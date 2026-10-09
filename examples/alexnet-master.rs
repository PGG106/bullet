#[path = "ti_inputs.rs"]
mod ti_inputs;

use std::sync::Arc;
use std::{cell::RefCell, mem::MaybeUninit};

use bullet_lib::{
    game::{
        inputs::{ChessBucketsMirrored, SparseInputType, get_num_buckets},
        outputs::OutputBuckets,
    },
    trainer::schedule::{
        lr::{self, LrScheduler},
        wdl,
    },
    value::{
        loader::SfBinpackLoader,
        save::{save_to_checkpoint, write_losses},
    },
    wdl::WdlScheduler,
};
use bullet_trainer::{
    model::{
        DenseInput, InitSettings, ModelDefinition, ModelEvaluator, ModelInputs, ModelInputsMapper, ModelWeights,
        SavedFormat, Shape, SparseInput,
    },
    optimiser::{
        Optimiser,
        adam::{AdamW, AdamWParams},
    },
    reader::ReadMapLoader,
    run::{DefaultDevice, HostPool, TrainingSchedule, TrainingSteps, train},
};
use rand::{
    Rng, SeedableRng,
    distr::{Bernoulli, Distribution},
    rng,
    rngs::StdRng,
};
use sfbinpack::TrainingDataEntry;
use sfbinpack::chess::attacks;
use sfbinpack::chess::bitboard::Bitboard;
use sfbinpack::chess::r#move::Move;
use sfbinpack::chess::r#move::MoveType;
use sfbinpack::chess::piecetype::PieceType;
use sfbinpack::chess::position::Position;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

const NET_NAME: &str = "mostdata-ti";
const READ_BUF_MB: usize = 8192;
const READ_THREADS: usize = 8;
const MAP_THREADS: u8 = 8;
const SAVE_RATE: usize = 80;
const CHECKPOINT_PATH: &str = "checkpoints\\fixedwdl-stage1-800";
const STAGE1_DATA_PATHS: [&str; 4] = [
    "data/master.binpack",
    "data/test79-2022-03-mar-16tb7p.v6-dd.binpack",
    "data/test79-2022-04-apr-16tb7p.v6-dd.binpack",
    "data/test80-2024-01-jan-2tb7p.min-v2.v6.relabel.binpack"];
const STAGE2_DATA_PATHS: [&str; 2] = ["data/master.binpack", "data/test79-2022-03-mar-16tb7p.v6-dd.binpack"];
const RUN_STAGE2: bool = false;

const L1: usize = 512;
const CLIP: f32 = 1.98;
const L2: usize = 16;
const L3: usize = 32;

const EVAL_SCALE: f32 = 362.0;
const QA: i16 = 255;
const QB: i16 = 64;

const STAGE1_SUPERBATCHES: usize = 800;
const STAGE2_START_SUPERBATCH: usize = 801;
const STAGE2_END_SUPERBATCH: usize = 1000;
const STAGE1_INITIAL_LR: f32 = 0.001;
const STAGE1_FINAL_LR: f32 = STAGE1_INITIAL_LR * 0.3 * 0.3 * 0.3 * 0.3 * 0.3 * 0.3 * 0.3;

#[derive(Clone, Copy, Default)]
pub struct CJBucket;
impl OutputBuckets<bulletformat::ChessBoard> for CJBucket {
    const BUCKETS: usize = 8;

    fn bucket(&self, pos: &bulletformat::ChessBoard) -> u8 {
        let pc_count = pos.occ().count_ones();
        ((63 - pc_count) * (32 - pc_count) / 225).min(7) as u8
    }
}

fn get_wdl(v: i16, pos: &sfbinpack::chess::position::Position) -> (f64, f64, f64) {
    let m = (pos.ply().min(240) as f64) / 64.0;

    // Coefficients from Stockfish WDL model
    const AS: [f64; 4] = [-3.68389304, 30.07065921, -60.52878723, 149.53378557];
    const BS: [f64; 4] = [-2.0181857, 15.85685038, -29.83452023, 47.59078827];

    let a = (((AS[0] * m + AS[1]) * m + AS[2]) * m) + AS[3];
    let mut b = (((BS[0] * m + BS[1]) * m + BS[2]) * m) + BS[3];

    b *= 1.5;

    let x = ((100.0 * v as f64) / 208.0).clamp(-2000.0, 2000.0);
    let w = 1.0 / (1.0 + ((a - x) / b).exp());
    let l = 1.0 / (1.0 + ((a + x) / b).exp());
    let d = 1.0 - w - l;

    (w, d, l)
}

thread_local! {
    static RNG: RefCell<StdRng> = RefCell::new(StdRng::seed_from_u64(42));
}

fn rng_keep(prob: f64) -> bool {
    RNG.with(|rng| {
        let distrib = Bernoulli::new(prob.clamp(0.0, 1.0)).unwrap();
        distrib.sample(&mut *rng.borrow_mut())
    })
}

fn shouldkeep(result: i16, v: i16, pos: &sfbinpack::chess::position::Position) -> bool {
    let (w, d, l) = get_wdl(v, pos);

    let keep_prob = if result > 0 {
        w
    } else if result < 0 {
        l
    } else {
        d
    };

    RNG.with(|rng| {
        let distrib = Bernoulli::new(keep_prob.clamp(0.0, 1.0)).unwrap();
        distrib.sample(&mut *rng.borrow_mut())
    })
}

fn piece_count_acceptance(pos: &sfbinpack::chess::position::Position) -> f64 {
    #[rustfmt::skip]
    const DESIRED_DISTRIBUTION: [f64; 33] = [
        0.018411966423, 0.020641545085, 0.022727271053,
        0.024669162740, 0.026467201733, 0.028121406444,
        0.029631758462, 0.030998276198, 0.032220941240,
        0.033299772000, 0.034234750067, 0.035025893853,
        0.035673184944, 0.036176641754, 0.036536245870,
        0.036752015705, 0.036823932846, 0.036752015705,
        0.036536245870, 0.036176641754, 0.035673184944,
        0.035025893853, 0.034234750067, 0.033299772000,
        0.032220941240, 0.030998276198, 0.029631758462,
        0.028121406444, 0.026467201733, 0.024669162740,
        0.022727271053, 0.020641545085, 0.018411966423,
    ];

    static PIECE_COUNT_STATS: [AtomicU64; 33] = {
        let mut arr: [std::mem::MaybeUninit<AtomicU64>; 33] = [const { MaybeUninit::uninit() }; 33];
        let mut i = 0;
        while i < 33 {
            arr[i].write(AtomicU64::new(0));
            i += 1;
        }
        unsafe { std::mem::transmute::<_, [AtomicU64; 33]>(arr) }
    };
    static PIECE_COUNT_TOTAL: AtomicU64 = AtomicU64::new(0);

    let pc = pos.occupied().count() as usize;
    let count = PIECE_COUNT_STATS[pc].fetch_add(1, Ordering::Relaxed) + 1;
    let total = PIECE_COUNT_TOTAL.fetch_add(1, Ordering::Relaxed) + 1;
    let frequency = count as f64 / total as f64;

    // Calculate the acceptance probability for this piece count
    let acceptance = 0.5 * DESIRED_DISTRIBUTION[pc] / frequency;
    acceptance.clamp(0., 1.)
}

fn skip_piececount(pos: &sfbinpack::chess::position::Position) -> bool {
    let mut rng = rng();
    rng.random_bool(piece_count_acceptance(pos))
}

const SEE_PIECE_VALUES: [i32; 7] = [100, 300, 330, 500, 900, 20000, 0]; // P, N, B, R, Q, K, None

pub fn estimated_see(pos: &Position, m: Move) -> i32 {
    // initially take the value of the thing on the target square
    let captured = pos.piece_at(m.to());
    let mut value = if captured.piece_type() == PieceType::None {
        0
    } else {
        SEE_PIECE_VALUES[captured.piece_type().ordinal() as usize]
    };

    if m.mtype() == MoveType::Promotion {
        // if it's a promo, swap a pawn for the promoted piece type
        let promo = m.promoted_piece().piece_type();
        value += SEE_PIECE_VALUES[promo.ordinal() as usize] - SEE_PIECE_VALUES[0];
    } else if m.mtype() == MoveType::EnPassant {
        // for e.p. we will miss a pawn because the target square is empty
        value = SEE_PIECE_VALUES[0];
    }

    value
}

pub fn static_exchange_eval(pos: &Position, m: Move, threshold: i32) -> bool {
    let from = m.from();
    let to = m.to();

    let mut next_victim = if m.mtype() == MoveType::Promotion {
        m.promoted_piece().piece_type()
    } else {
        pos.piece_at(from).piece_type()
    };

    let mut balance = estimated_see(pos, m) - threshold;

    // if the best case fails, don't bother doing the full search.
    if balance < 0 {
        return false;
    }

    // worst case is losing the piece
    balance -= SEE_PIECE_VALUES[next_victim.ordinal() as usize];

    // if the worst case passes, we can return true immediately.
    if balance >= 0 {
        return true;
    }

    let mut occupied = pos.occupied();
    occupied.set(from.index(), false);
    occupied.set(to.index(), true);

    if m.mtype() == MoveType::EnPassant {
        occupied.set(to.index() ^ 8, false);
    }

    // after the move, it's the opponent's turn.
    let mut colour = !pos.side_to_move();

    let get_attackers = |sq, occ: Bitboard| {
        (attacks::pawn(sfbinpack::chess::color::Color::White, sq)
            & pos.pieces_bb_color(sfbinpack::chess::color::Color::Black, PieceType::Pawn)
            | attacks::pawn(sfbinpack::chess::color::Color::Black, sq)
                & pos.pieces_bb_color(sfbinpack::chess::color::Color::White, PieceType::Pawn)
            | attacks::knight(sq) & pos.pieces_bb_type(PieceType::Knight)
            | attacks::king(sq) & pos.pieces_bb_type(PieceType::King)
            | attacks::bishop(sq, occ) & (pos.pieces_bb_type(PieceType::Bishop) | pos.pieces_bb_type(PieceType::Queen))
            | attacks::rook(sq, occ) & (pos.pieces_bb_type(PieceType::Rook) | pos.pieces_bb_type(PieceType::Queen)))
            & occ
    };

    let mut attackers = get_attackers(to, occupied);

    loop {
        let my_attackers = attackers & pos.pieces_bb(colour);
        if my_attackers.bits() == 0 {
            break;
        }

        // find cheapest attacker
        for victim_idx in 0..6 {
            let victim = PieceType::from_ordinal(victim_idx as u8);
            if (my_attackers & pos.pieces_bb_type(victim)).bits() != 0 {
                next_victim = victim;
                break;
            }
        }

        let lsb = (my_attackers & pos.pieces_bb_type(next_victim)).lsb();
        occupied.set(lsb.index(), false);

        // diagonal moves reveal bishops and queens:
        if next_victim == PieceType::Pawn || next_victim == PieceType::Bishop || next_victim == PieceType::Queen {
            attackers |= attacks::bishop(to, occupied)
                & (pos.pieces_bb_type(PieceType::Bishop) | pos.pieces_bb_type(PieceType::Queen));
        }

        // orthogonal moves reveal rooks and queens:
        if next_victim == PieceType::Rook || next_victim == PieceType::Queen {
            attackers |= attacks::rook(to, occupied)
                & (pos.pieces_bb_type(PieceType::Rook) | pos.pieces_bb_type(PieceType::Queen));
        }

        attackers = attackers & occupied;

        colour = !colour;

        balance = -balance - 1 - SEE_PIECE_VALUES[next_victim.ordinal() as usize];

        if balance >= 0 {
            if next_victim == PieceType::King && (attackers & pos.pieces_bb(colour)).bits() != 0 {
                colour = !colour;
            }
            break;
        }
    }

    // the side that is to move after loop exit is the loser.
    pos.side_to_move() != colour
}

// currently does nothing
const NUM_OUTPUT_BUCKETS: usize = 8;
#[rustfmt::skip]
    const BUCKET_LAYOUT: [usize; 32] = [
        0,  1,  2,  3,
        4,  5,  6,  7,
        8,  9, 10, 11,
        8,  9, 10, 11,
        12, 12, 13, 13,
        12, 12, 13, 13,
        14, 14, 15, 15,
        14, 14, 15, 15
    ];

const NUM_INPUT_BUCKETS: usize = get_num_buckets(&BUCKET_LAYOUT);

fn build_bbs(pos: &bulletformat::ChessBoard) -> [u64; 8] {
    let mut bbs = [0u64; 8];

    for (pc, sq) in pos.into_iter() {
        let bit = 1 << sq;
        bbs[usize::from(pc & 8 > 0)] |= bit;
        bbs[2 + usize::from(pc & 7)] |= bit;
    }

    bbs
}

#[derive(Clone)]
struct ThreatInputs {
    threats: Arc<ti_inputs::Threats>,
}

impl ThreatInputs {
    fn new() -> Self {
        Self { threats: Arc::new(ti_inputs::Threats::new()) }
    }

    fn num_inputs(&self) -> usize {
        self.threats.num_inputs()
    }

    fn max_active(&self) -> usize {
        self.threats.max_active()
    }

    fn map_features(&self, pos: &bulletformat::ChessBoard, on_stm: impl FnMut(usize), on_ntm: impl FnMut(usize)) {
        let bbs = build_bbs(pos);
        self.threats.map(bbs, on_stm, on_ntm);
    }
}

// stm psqt, ntm psqt, stm threats, ntm threats, output buckets, targets
type InputTy = (((((SparseInput, SparseInput), SparseInput), SparseInput), SparseInput), DenseInput<f32>);

fn make_inputs_mapper(
    inputs: &ModelInputs<InputTy>,
    feature_getter: ChessBucketsMirrored,
    threats: ThreatInputs,
    output_buckets: CJBucket,
    wdl: impl WdlScheduler,
) -> ModelInputsMapper<bulletformat::ChessBoard> {
    ModelInputsMapper::build(inputs, move |pos, step, (((((stm, ntm), stm_t), ntm_t), bucket), target)| {
        let mut count = 0;
        feature_getter.map_features(pos, |stm_feature, ntm_feature| {
            stm[count] = stm_feature.try_into().unwrap();
            ntm[count] = ntm_feature.try_into().unwrap();
            count += 1;
        });

        assert!(count <= feature_getter.max_active(), "More inputs provided than the specified maximum!");
        if count < feature_getter.max_active() {
            stm[count] = -1;
            ntm[count] = -1;
        }

        let mut stm_cnt = 0;
        let mut ntm_cnt = 0;
        threats.map_features(
            pos,
            |f| {
                stm_t[stm_cnt] = f.try_into().unwrap();
                stm_cnt += 1;
            },
            |f| {
                ntm_t[ntm_cnt] = f.try_into().unwrap();
                ntm_cnt += 1;
            },
        );

        assert!(
            stm_cnt <= threats.max_active() && ntm_cnt <= threats.max_active(),
            "More threats provided than the specified maximum!"
        );
        if stm_cnt < threats.max_active() {
            stm_t[stm_cnt] = -1;
        }
        if ntm_cnt < threats.max_active() {
            ntm_t[ntm_cnt] = -1;
        }

        bucket[0] = i32::from(output_buckets.bucket(pos));

        let result = f32::from(pos.result) / 2.0;
        let score = 1.0 / (1.0 + (f32::from(-pos.score) / EVAL_SCALE).exp());
        let lambda = wdl.blend(step.batch(), step.superbatch(), step.final_superbatch());
        assert!((0.0..=1.0).contains(&lambda), "WDL proportion must be in [0, 1]");
        target[0] = lambda * result + (1.0 - lambda) * score;
    })
}

fn main() {
    let feature_getter = ChessBucketsMirrored::new(BUCKET_LAYOUT);
    let threats = ThreatInputs::new();
    let output_buckets = CJBucket;
    let inputs = ModelInputs::default()
        .add_sparse("stm", (feature_getter.num_inputs(), 1), feature_getter.max_active())
        .add_sparse("nstm", (feature_getter.num_inputs(), 1), feature_getter.max_active())
        .add_sparse("stm_threats", (threats.num_inputs(), 1), threats.max_active())
        .add_sparse("ntm_threats", (threats.num_inputs(), 1), threats.max_active())
        .add_sparse("buckets", (NUM_OUTPUT_BUCKETS, 1), 1)
        .add_dense("targets", (1, 1));

    let defn = ModelDefinition::build(
        &inputs,
        |builder, (((((stm_inputs, ntm_inputs), stm_threats), ntm_threats), output_buckets), target)| {
            let l0f = builder.new_weights("l0f", Shape::new(L1, 768), InitSettings::Zeroed);
            let expanded_factoriser = l0f.repeat(NUM_INPUT_BUCKETS);

            // PSQT weights only; l0t supplies the combined accumulator bias.
            let l0 = builder.new_weights(
                "l0w",
                Shape::new(L1, 768 * NUM_INPUT_BUCKETS),
                InitSettings::Normal { mean: 0.0, stdev: (2f32 / 32.0).sqrt() },
            );
            let l0 = (l0 + expanded_factoriser).clip_pass_through_grad(-CLIP, CLIP);

            // threat input weights, summed with the psqt part before the activation
            let l0t = builder.new_affine("l0t", threats.num_inputs(), L1);

            // output layer weights
            let l1 = builder.new_affine("l1", L1, NUM_OUTPUT_BUCKETS * L2);
            let l2 = builder.new_affine("l2", L2 * 2, NUM_OUTPUT_BUCKETS * L3);
            let l3 = builder.new_affine("l3", L3, NUM_OUTPUT_BUCKETS);

            let ft = |psqt, thr, start, end| {
                (l0.slice_rows(start, end).matmul(psqt) + l0t.slice(start, end).forward(thr)).crelu()
            };
            let stm_hidden = ft(stm_inputs, stm_threats, 0, L1 / 2) * ft(stm_inputs, stm_threats, L1 / 2, L1);
            let ntm_hidden = ft(ntm_inputs, ntm_threats, 0, L1 / 2) * ft(ntm_inputs, ntm_threats, L1 / 2, L1);

            let hl1 = stm_hidden.concat(ntm_hidden);

            let ones_l1_vec = builder.new_constant(Shape::new(1, L1), &[1.0 / L1 as f32; L1]);
            let l0_out_norm = ones_l1_vec.matmul(hl1);

            let l1_out = l1.forward(hl1).select(output_buckets);
            let hl2 = l1_out.concat(l1_out.abs_pow(2.0)).crelu();

            let hl3 = l2.forward(hl2).select(output_buckets).screlu();
            let l3_out = l3.forward(hl3).select(output_buckets);

            let loss = l3_out.sigmoid().power_error(target, 2.5);
            let loss = loss + 0.004 * l0_out_norm;

            (Some(loss.reduce_sum_batch()), vec![("output".to_string(), l3_out)])
        },
    );

    let weights = ModelWeights::new(&defn, 198273612);
    let device = DefaultDevice::new(0).unwrap();
    let mut evaluator = ModelEvaluator::new(&defn, device.clone()).unwrap();
    let mut optimiser = Optimiser::<_, AdamW<_>>::new(defn, weights, device.clone(), AdamWParams::default()).unwrap();
    let no_clipping = AdamWParams { min_weight: -128.0, max_weight: 128.0, ..Default::default() };

    optimiser.set_params_for_weight("l2w", no_clipping);
    optimiser.set_params_for_weight("l2b", no_clipping);
    optimiser.set_params_for_weight("l3w", no_clipping);
    optimiser.set_params_for_weight("l3b", no_clipping);

    let l0_clip = AdamWParams { min_weight: -CLIP / 2.0, max_weight: CLIP / 2.0, ..Default::default() };
    optimiser.set_params_for_weight("l0w", l0_clip);
    optimiser.set_params_for_weight("l0f", l0_clip);
    optimiser.set_params_for_weight("l0tw", l0_clip);

    let saved_format = vec![
        SavedFormat::id("l0w")
            .transform(|store, weights| {
                let factoriser = store.get("l0f").values.f32().repeat(NUM_INPUT_BUCKETS);
                weights.iter().zip(factoriser).map(|(a, b)| a + b).collect()
            })
            .round()
            .quantise::<i16>(QA),
        SavedFormat::id("l0tw").round().quantise::<i16>(QA),
        SavedFormat::id("l0tb").round().quantise::<i16>(QA),
        SavedFormat::id("l1w").round().quantise::<i8>(QB),
        SavedFormat::id("l1b"),
        SavedFormat::id("l2w"),
        SavedFormat::id("l2b"),
        SavedFormat::id("l3w"),
        SavedFormat::id("l3b"),
    ];

    // optimiser.load_from_checkpoint(&format!("{CHECKPOINT_PATH}\\optimiser_state")).unwrap();

    let mut run = |stage, start_superbatch, end_superbatch, lr_schedule, mapper, reader| {
        let error_record = RefCell::new(Vec::new());
        let mut loss_sum = 0.0;
        let mut ticks_since_last = 0.0;

        train(
            &mut optimiser,
            TrainingSchedule {
                steps: TrainingSteps {
                    batch_size: 16_384 * 8,
                    batches_per_superbatch: 6104 / 8,
                    start_superbatch,
                    end_superbatch,
                },
                lr_schedule,
                log_rate: 128,
            },
            ReadMapLoader::new(reader, mapper, MAP_THREADS),
            |_, step, error| {
                loss_sum += error;
                ticks_since_last += 1.0;

                if step.batch().is_multiple_of(32)
                    || (step.batches_per_superbatch() < 32 && step.batch() == step.batches_per_superbatch())
                {
                    let normalised_loss = loss_sum / f32::min(ticks_since_last, step.batches_per_superbatch() as f32);
                    error_record.borrow_mut().push((step.superbatch(), step.batch(), normalised_loss));
                    loss_sum = 0.0;
                    ticks_since_last = 0.0;
                }
            },
            |optimiser, step| {
                let superbatch = step.superbatch();
                if superbatch.is_multiple_of(SAVE_RATE) || superbatch == step.final_superbatch() {
                    let name = format!("{NET_NAME}-stage{stage}-{superbatch}");
                    let path = format!("checkpoints/{name}");
                    save_to_checkpoint(optimiser, &saved_format, &path);
                    write_losses(&format!("{path}/log.txt"), &error_record.borrow());
                    println!("Saved [{name}]");
                }
            },
        )
        .unwrap();
    };

    fn stage1_filter(entry: &TrainingDataEntry) -> bool {
        entry.ply >= 16
            && !entry.pos.is_checked(entry.pos.side_to_move())
            && entry.score.unsigned_abs() <= 25000
            && entry.mv.mtype() == MoveType::Normal
            && entry.pos.piece_at(entry.mv.to()).piece_type() == PieceType::None
            && shouldkeep(entry.result, entry.score, &entry.pos)
            && skip_piececount(&entry.pos)
    }

    let reader = SfBinpackLoader::new_concat_multiple(
        &STAGE1_DATA_PATHS,
        READ_BUF_MB,
        READ_THREADS,
        stage1_filter as fn(&TrainingDataEntry) -> bool,
    );
    let mapper = make_inputs_mapper(
        &inputs,
        feature_getter,
        threats.clone(),
        output_buckets,
        wdl::LinearWDL { start: 0.0, end: 0.15 },
    );
    run(
        1,
        1,
        STAGE1_SUPERBATCHES,
        lr::Warmup {
            inner: lr::CosineDecayLR {
                initial_lr: STAGE1_INITIAL_LR,
                final_lr: STAGE1_FINAL_LR,
                final_superbatch: STAGE1_SUPERBATCHES,
            },
            warmup_batches: 200,
        }
        .boxed(),
        mapper,
        reader,
    );

    if RUN_STAGE2 {
        fn stage2_filter(entry: &TrainingDataEntry) -> bool {
            entry.ply >= 28
                && !entry.pos.is_checked(entry.pos.side_to_move())
                && entry.score.unsigned_abs() <= 20000
                && entry.mv.mtype() == MoveType::Normal
                && entry.pos.piece_at(entry.mv.to()).piece_type() == PieceType::None
                && shouldkeep(entry.result, entry.score, &entry.pos)
                && skip_piececount(&entry.pos)
        }

        let reader = SfBinpackLoader::new_concat_multiple(
            &STAGE2_DATA_PATHS,
            READ_BUF_MB,
            READ_THREADS,
            stage2_filter as fn(&TrainingDataEntry) -> bool,
        );
        let mapper = make_inputs_mapper(
            &inputs,
            feature_getter,
            threats.clone(),
            output_buckets,
            wdl::ConstantWDL { value: 0.15 },
        );
        run(
            2,
            STAGE2_START_SUPERBATCH,
            STAGE2_END_SUPERBATCH,
            lr::Warmup {
                inner: lr::CosineDecayLR {
                    initial_lr: STAGE1_INITIAL_LR * 0.1,
                    final_lr: STAGE1_FINAL_LR * 0.5,
                    final_superbatch: STAGE2_END_SUPERBATCH,
                },
                warmup_batches: 10,
            }
            .boxed(),
            mapper,
            reader,
        );
    }

    evaluator.load_device_weights(optimiser.weights()).unwrap();
    let evaluator_mapper = make_inputs_mapper(
        &inputs,
        feature_getter,
        threats.clone(),
        output_buckets,
        wdl::ConstantWDL { value: 0.0 },
    );
    let pool = HostPool::new(device.clone());

    // Keep this output with the checkpoint: it is the reference for the engine eval-matching test.
    for fen in [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/P2P2PP/rq2Q1R1K w kq - 0 2",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1",
        "3N4/b2R2p1/3q3r/6P1/4k1nQ/7B/8/K7 w - - 0 1",
        "k2B1Q1q/8/b7/4p3/3Pr3/1N5R/2n5/1K6 w - - 0 1",
        "1B3q2/8/r5n1/8/Rp1N1PQ1/8/4bk2/2K5 w - - 0 1",
        "8/5NR1/5q1b/8/7p/3P2B1/6Q1/1k1K1n1r w - - 0 1",
        "8/8/6r1/4B3/3Q3p/N1nq4/5RP1/b3K2k b - - 0 1",
        "3qn2Q/1R6/8/1N3b1p/4B3/1kP5/r7/5K2 b - - 0 1",
        "3rBR2/2qQ1p2/N7/2P2b2/6n1/k7/8/6K1 b - - 0 1",
        "k7/8/p1rB1q2/7Q/4R3/2N2n2/7P/6bK b - - 0 1",
        "2n2Rr1/Bk5p/N7/2Q3q1/b7/8/KP6/8 w - - 0 1",
        "8/Q6r/3qR1P1/b4p2/k7/3B4/1KN2n2/8 b - - 0 1",
    ] {
        let pos = format!("{fen} | 0 | 0.0").parse().unwrap();
        let inputs = evaluator_mapper.map(&pool, &[pos], Default::default(), 1).unwrap().to_device(&device).unwrap();
        let output = evaluator.evaluate(&inputs).unwrap().get("output").unwrap();
        let [eval] = output.to_host().unwrap().f32()[..] else { panic!() };
        println!("FEN: {fen}");
        println!("EVAL: {}", EVAL_SCALE * eval);
    }
}