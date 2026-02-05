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

fn do_skip(prob: f64) -> bool {
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
    const HL_SIZE: usize = 1536;
    const CLIP: f32 = 1.98;
    let l2 = 16;
    let l3 = 32;
    let name = "foresight2-l0reg";
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
            // input layer factoriser
            let l0f = builder.new_weights("l0f", Shape::new(HL_SIZE, 768), InitSettings::Zeroed);
            let expanded_factoriser = l0f.repeat(NUM_INPUT_BUCKETS);

            // input layer weights
            let mut l0 = builder.new_affine("l0", 768 * NUM_INPUT_BUCKETS, HL_SIZE);
            l0.init_with_effective_input_size(32);
            l0.weights = (l0.weights + expanded_factoriser).clip_pass_through_grad(-CLIP, CLIP);

            // layerstack weights
            let l1 = builder.new_affine("l1", HL_SIZE, NUM_OUTPUT_BUCKETS * l2);
            let l2 = builder.new_affine("l2", l2, NUM_OUTPUT_BUCKETS * l3);
            let l3 = builder.new_affine("l3", l3, NUM_OUTPUT_BUCKETS);

            // inference
            let stm_hidden = l0.forward(stm_inputs).crelu().pairwise_mul();
            let ntm_hidden = l0.forward(ntm_inputs).crelu().pairwise_mul();
            let l0_out = stm_hidden.concat(ntm_hidden);

            let ones_l1_vec = builder.new_constant(Shape::new(1, HL_SIZE), &[1.0 / HL_SIZE as f32; HL_SIZE]);
            let l0_out_norm = ones_l1_vec.matmul(l0_out);

            let hl1 = l1.forward(l0_out).select(output_buckets);
            let hl2 = l1.forward(hl1).select(output_buckets).screlu();
            let hl3 = l2.forward(hl2).select(output_buckets).screlu();
            let l3_out = l3.forward(hl3).select(output_buckets);

            let loss = l3_out.sigmoid().squared_error(target);
            let loss = loss + 0.005 * l0_out_norm;
            (l3_out, loss)
        });

    let no_clipping = AdamWParams { min_weight: -128.0, max_weight: 128.0, ..Default::default() };

    trainer.optimiser.set_params_for_weight("l2w", no_clipping);
    trainer.optimiser.set_params_for_weight("l2b", no_clipping);
    trainer.optimiser.set_params_for_weight("l3w", no_clipping);
    trainer.optimiser.set_params_for_weight("l3b", no_clipping);

    let wdl_scheduler = wdl::ConstantWDL { value: 0.0 };

    let lr_scheduler = lr::Warmup {
        inner: lr::CosineDecayLR { initial_lr: s1_initial_lr, final_lr: s1_final_lr, final_superbatch: STAGE1_SB },
        warmup_batches: 800,
    };

    let schedule = TrainingSchedule {
        net_id: (name.to_owned() + "-stage1").to_string(),
        eval_scale: 362.0,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
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
                && entry.score.unsigned_abs() <= 10000
                && entry.mv.mtype() == MoveType::Normal
                && entry.pos.piece_at(entry.mv.to()).piece_type() == PieceType::None
                && shouldkeep(entry.result, entry.score, &entry.pos)
        }
        SfBinpackLoader::new_concat_multiple(&file_path, buffer_size_mb, threads, filter)
    };

    /*
        let dataloader = {
          let file_path = "data/monty.binpack";
          let buffer_size_mb = 4096;
          let threads = 6;
          fn filter(pos: &Position, best_move: Move, score: i16, _result: f32) -> bool {
              pos.fullm() >= 8
                  && score.unsigned_abs() <= 10000
                  && !best_move.is_capture()
                  && !best_move.is_promo()
                  && !pos.in_check()
          }

          loader::MontyBinpackLoader::new(file_path, buffer_size_mb, threads, filter)
      };
    */

    // loading directly from a `BulletFormat` file
    //let dataloader = loader::DirectSequentialDataLoader::new(&["data/baseline.data"]);

    // trainer.load_from_checkpoint("checkpoints\\moarlayers-wdlskip2-800");

    //trainer.save_to_checkpoint("checkpoints\\fixed-shit");

    // trainer.load_from_checkpoint("checkpoints\\foresight2-stage1-560");

    trainer.run(&schedule, &settings, &dataloader);

    // Stage 2

    // same LR and WDL

    // start at sb 700
    let schedule = TrainingSchedule {
        net_id: (name.to_owned() + "-stage2").to_string(),
        eval_scale: 362.0,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
            start_superbatch: 700,
            end_superbatch: STAGE1_SB,
        },
        wdl_scheduler,
        lr_scheduler,
        save_rate: 80,
    };

    // use different binpack set
    let dataset_path = ["data/master.binpack", "data/t60-2020.binpack"];

    let dataloader = {
        let file_path = dataset_path;
        let buffer_size_mb = 4096;
        let threads = 4;
        fn filter(entry: &TrainingDataEntry) -> bool {
            entry.ply >= 28
                && !entry.pos.is_checked(entry.pos.side_to_move())
                && entry.score.unsigned_abs() <= 10000
                && entry.mv.mtype() == MoveType::Normal
                && entry.pos.piece_at(entry.mv.to()).piece_type() == PieceType::None
                && shouldkeep(entry.result, entry.score, &entry.pos)
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
