use std::{cell::RefCell, mem::MaybeUninit};

use bullet_lib::{
    game::{
        inputs::{ChessBucketsMirrored, get_num_buckets},
        outputs::MaterialCount,
    },
    nn::{
        InitSettings, Shape,
        optimiser::{AdamW, AdamWParams},
    },
    trainer::{
        save::SavedFormat,
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::LocalSettings,
    },
    value::{ValueTrainerBuilder, loader::DirectSequentialDataLoader},
};

use bullet_lib::game::outputs::OutputBuckets;
use bullet_lib::value::loader::SfBinpackLoader;
use rand::{
    Rng, SeedableRng,
    distr::{Bernoulli, Distribution},
    rng,
    rngs::StdRng,
};
use sfbinpack::TrainingDataEntry;
use sfbinpack::chess::r#move::MoveType;
use sfbinpack::chess::piecetype::PieceType;
use sfbinpack::chess::position::Position;
use sfbinpack::chess::r#move::Move;
use sfbinpack::chess::bitboard::Bitboard;
use sfbinpack::chess::attacks;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::{AtomicUsize, Ordering};

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
        occupied.set((to.index() ^ 8), false);
    }

    // after the move, it's the opponent's turn.
    let mut colour = !pos.side_to_move();

    let get_attackers = |sq, occ: Bitboard| {
        (attacks::pawn(sfbinpack::chess::color::Color::White, sq) & pos.pieces_bb_color(sfbinpack::chess::color::Color::Black, PieceType::Pawn)
            | attacks::pawn(sfbinpack::chess::color::Color::Black, sq) & pos.pieces_bb_color(sfbinpack::chess::color::Color::White, PieceType::Pawn)
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
        if next_victim == PieceType::Pawn
            || next_victim == PieceType::Bishop
            || next_victim == PieceType::Queen
        {
            attackers |= attacks::bishop(to, occupied) & (pos.pieces_bb_type(PieceType::Bishop) | pos.pieces_bb_type(PieceType::Queen));
        }

        // orthogonal moves reveal rooks and queens:
        if next_victim == PieceType::Rook || next_victim == PieceType::Queen {
            attackers |= attacks::rook(to, occupied) & (pos.pieces_bb_type(PieceType::Rook) | pos.pieces_bb_type(PieceType::Queen));
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

fn main() {
    // hyperparams to fiddle with
    const L1: usize = 1536;
    const CLIP: f32 = 1.98;
    const L2: usize = 16;
    const L3: usize = 32;
    let name = "fixedwdl";
    let dataset_path = ["data/master.binpack"];
    let s1_initial_lr = 0.001;
    let s1_final_lr = 0.001 * 0.3 * 0.3 * 0.3 * 0.3 * 0.3 * 0.3 * 0.3;
    const STAGE1_SB: usize = 800;
    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .inputs(ChessBucketsMirrored::new(BUCKET_LAYOUT))
        .output_buckets(CJBucket)
        .save_format(&[
            SavedFormat::id("l0f"),
            SavedFormat::id("l0w"),
            SavedFormat::id("l0b"),
            SavedFormat::id("l1w"),
            SavedFormat::id("l1b"),
            SavedFormat::id("l2w"),
            SavedFormat::id("l2b"),
            SavedFormat::id("l3w"),
            SavedFormat::id("l3b"),
        ])
        .build_custom(|builder, (stm_inputs, ntm_inputs, output_buckets), target| {
            let l0f = builder.new_weights("l0f", Shape::new(L1, 768), InitSettings::Zeroed);
            let expanded_factoriser = l0f.repeat(NUM_INPUT_BUCKETS);

            // input layer weights
            let mut l0 = builder.new_affine("l0", 768 * NUM_INPUT_BUCKETS, L1);
            l0.init_with_effective_input_size(32);
            l0.weights = (l0.weights + expanded_factoriser).clip_pass_through_grad(-CLIP, CLIP);

            // output layer weights
            let l1 = builder.new_affine("l1", L1, NUM_OUTPUT_BUCKETS * L2);
            let l2 = builder.new_affine("l2", L2 * 2, NUM_OUTPUT_BUCKETS * L3);
            let l3 = builder.new_affine("l3", L3, NUM_OUTPUT_BUCKETS);

            // inference
            let stm_hidden = l0.forward(stm_inputs).crelu().pairwise_mul();
            let ntm_hidden = l0.forward(ntm_inputs).crelu().pairwise_mul();
            let hl1 = stm_hidden.concat(ntm_hidden);

            let ones_l1_vec = builder.new_constant(Shape::new(1, L1), &[1.0 / L1 as f32; L1]);
            let l0_out_norm = ones_l1_vec.matmul(hl1);

            let l1_out = l1.forward(hl1).select(output_buckets);
            let hl2 = l1_out.concat(l1_out.abs_pow(2.0)).crelu();

            let hl3 = l2.forward(hl2).select(output_buckets).screlu();
            let l3_out = l3.forward(hl3).select(output_buckets);

            let loss = l3_out.sigmoid().power_error(target, 2.5);
            let loss = loss + 0.004 * l0_out_norm;

            return (l3_out, loss);
        });


    let no_clipping = AdamWParams { min_weight: -128.0, max_weight: 128.0, ..Default::default() };

    trainer.optimiser.set_params_for_weight("l2w", no_clipping);
    trainer.optimiser.set_params_for_weight("l2b", no_clipping);
    trainer.optimiser.set_params_for_weight("l3w", no_clipping);
    trainer.optimiser.set_params_for_weight("l3b", no_clipping);

    let wdl_scheduler = wdl::LinearWDL {start : 0.0,  end:0.15};

    let lr_scheduler = lr::Warmup {
        inner: lr::CosineDecayLR { initial_lr: s1_initial_lr, final_lr: s1_final_lr, final_superbatch: STAGE1_SB },
        warmup_batches: 200,
    };

    let schedule = TrainingSchedule {
        net_id: (name.to_owned() + "-stage1").to_string(),
        eval_scale: 362.0,
        steps: TrainingSteps {
            batch_size: 16_384 * 8 ,
            batches_per_superbatch: 6104 / 8,
            start_superbatch: 1,
            end_superbatch: STAGE1_SB,
        },
        wdl_scheduler: wdl_scheduler.clone(),
        lr_scheduler: lr_scheduler.clone(),
        save_rate: 80,
    };

    let settings = LocalSettings { threads: 2, test_set: None, output_directory: "checkpoints", batch_queue_size: 64 };

    // loading from a SF binpack
    let dataloader = {
        let file_path = dataset_path;
        let buffer_size_mb = 4096;
        let threads = 4;
        fn filter(entry: &TrainingDataEntry) -> bool {
            entry.ply >= 28
                && !entry.pos.is_checked(entry.pos.side_to_move())
                && entry.score.unsigned_abs() <= 20000
                && (entry.mv.mtype() == MoveType::Normal)
                && entry.pos.piece_at(entry.mv.to()).piece_type() == PieceType::None
                && shouldkeep(entry.result, entry.score, &entry.pos)
                && skip_piececount(&entry.pos)
        }
        SfBinpackLoader::new_concat_multiple(&file_path, buffer_size_mb, threads, filter)
    };

    trainer.load_from_checkpoint("checkpoints\\fixedwdl-stage1-800");

    //trainer.save_to_checkpoint("checkpoints\\fixed-shit");


    trainer.run(&schedule, &settings, &dataloader);

    // Stage 2

    let wdl_scheduler = wdl::ConstantWDL {value:0.15};

    let lr_scheduler = lr::Warmup {
        inner: lr::CosineDecayLR { initial_lr: s1_initial_lr * 0.1, final_lr: s1_final_lr * 0.25, final_superbatch: 1000},
        warmup_batches: 10,
    };

    // start at sb 700
    let schedule = TrainingSchedule {
        net_id: (name.to_owned() + "-stage2").to_string(),
        eval_scale: 362.0,
        steps: TrainingSteps {
            batch_size: 16_384 * 8 ,
            batches_per_superbatch: 6104 / 8,
            start_superbatch: 801,
            end_superbatch: 1000,
        },
        wdl_scheduler,
        lr_scheduler,
        save_rate: 80,
    };

    // use different binpack set
    let dataset_path = [
    "data/test78-junjulaug2022-16tb7p-eval-filt-v2-d6.binpack"];

    let dataloader = {
        let file_path = dataset_path;
        let buffer_size_mb = 4096;
        let threads = 4;
        fn filter(entry: &TrainingDataEntry) -> bool {
            entry.ply >= 28
                && !entry.pos.is_checked(entry.pos.side_to_move())
                && entry.score.unsigned_abs() <= 20000
                && entry.mv.mtype() == MoveType::Normal
                && (entry.pos.piece_at(entry.mv.to()).piece_type() == PieceType::None)
                && shouldkeep(entry.result, entry.score, &entry.pos)
                && skip_piececount(&entry.pos)
        }
        SfBinpackLoader::new_concat_multiple(&file_path, buffer_size_mb, threads, filter)
    };

    trainer.run(&schedule, &settings, &dataloader);

    for fen in [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    ] {
        let eval = trainer.eval(fen);
        println!("FEN: {fen}");
        println!("EVAL: {}", 362.0 * eval);
    }
}
