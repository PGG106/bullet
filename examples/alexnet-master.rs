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
use rand::{Rng, SeedableRng, rngs::StdRng};
use sfbinpack::TrainingDataEntry;
use sfbinpack::chess::r#move::MoveType;
use sfbinpack::chess::piecetype::PieceType;

#[derive(Clone, Copy, Default)]
pub struct CJBucket;
impl OutputBuckets<bulletformat::ChessBoard> for CJBucket {
    const BUCKETS: usize = 8;

    fn bucket(&self, pos: &bulletformat::ChessBoard) -> u8 {
        let pc_count = pos.occ().count_ones();
        ((63 - pc_count) * (32 - pc_count) / 225).min(7) as u8
    }
}

fn phase(pos: &sfbinpack::chess::position::Position) -> u8 {
    let pawns = pos.pieces_bb_type(PieceType::Pawn).count();
    let knights = pos.pieces_bb_type(PieceType::Knight).count();
    let bishops = pos.pieces_bb_type(PieceType::Bishop).count();
    let rooks = pos.pieces_bb_type(PieceType::Rook).count();
    let queens = pos.pieces_bb_type(PieceType::Queen).count();

    (pawns + 3 * knights + 3 * bishops + 5 * rooks + 9 * queens) as u8
}

fn win_rate_params(pos: &sfbinpack::chess::position::Position) -> (f64, f64) {
    let material = phase(pos);

    let m = (material.clamp(17, 78) as f64) / 58.0;

    // Coefficients from Stockfish WDL model
    const AS: [f64; 4] = [-13.50030198, 40.92780883, -36.82753545, 386.83004070];
    const BS: [f64; 4] = [96.53354896, -165.79058388, 90.89679019, 49.29561889];

    let a = (((AS[0] * m + AS[1]) * m + AS[2]) * m) + AS[3];
    let b = (((BS[0] * m + BS[1]) * m + BS[2]) * m) + BS[3];

    (a, b)
}

fn win_rate_model(v: i16, pos: &sfbinpack::chess::position::Position) -> u16 {
    let (a, b) = win_rate_params(pos);

    // Return the win rate in per mille units, rounded to the nearest integer.
    let win_rate = (0.5 + 1000.0 / (1.0 + ((a - v as f64) / b).exp())).round() as u16;

    win_rate
}

fn get_wdl(v: i16, pos: &sfbinpack::chess::position::Position) -> (u16, u16, u16) {
    let win_rate = win_rate_model(v, pos);
    let loss_rate = win_rate_model(-v, pos);
    let draw_rate = 1000 - win_rate - loss_rate;

    (win_rate, draw_rate, loss_rate)
}

fn shouldkeep(result: i16, v: i16, pos: &sfbinpack::chess::position::Position) -> bool {
    let wdl = get_wdl(v, pos);
    let mut keep_prob = 0;
    // 1, 0, -1 for white win, draw, white loss respectively.
    if result == 1 {
        keep_prob = wdl.0;
    }
    if result == 0 {
        keep_prob = wdl.1;
    }
    if result == -1 {
        keep_prob = wdl.2;
    }

    // rng?
    let seed: [u8; 32] = [1; 32];
    let mut rng = StdRng::from_seed(seed);

    // Generate some random numbers
    let random_number: u16 = rng.random_range(0..1000);

    random_number > 1000 - keep_prob
}

fn main() {
    // hyperparams to fiddle with
    let hl_size = 1536;
    const CLIP: f32 = 1.98;
    let l2 = 16;
    let l3 = 32;
    let dataset_path = "data/master.binpack";
    let s1_initial_lr = 0.001;
    let s1_final_lr = 0.001 * 0.3 * 0.3 * 0.3 * 0.3 * 0.3 * 0.3 * 0.3;
    const STAGE1_SB: usize = 800;
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
        .loss_fn(|output, target| output.sigmoid().power_error(target, 2.5))
        .build(|builder, stm_inputs, ntm_inputs, output_buckets| {
            // input layer factoriser
            let l0f = builder.new_weights("l0f", Shape::new(hl_size, 768), InitSettings::Zeroed);
            let expanded_factoriser = l0f.repeat(NUM_INPUT_BUCKETS);

            // input layer weights
            let mut l0 = builder.new_affine("l0", 768 * NUM_INPUT_BUCKETS, hl_size);
            l0.init_with_effective_input_size(32);
            l0.weights = (l0.weights + expanded_factoriser).clip_pass_through_grad(-CLIP, CLIP);

            // layerstack weights
            let l1 = builder.new_affine("l1", hl_size, NUM_OUTPUT_BUCKETS * l2);
            let l2 = builder.new_affine("l2", l2, NUM_OUTPUT_BUCKETS * l3);
            let l3 = builder.new_affine("l3", l3, NUM_OUTPUT_BUCKETS);

            // inference
            let stm_hidden = l0.forward(stm_inputs).crelu().pairwise_mul();
            let ntm_hidden = l0.forward(ntm_inputs).crelu().pairwise_mul();
            let hl1 = stm_hidden.concat(ntm_hidden);
            let hl2 = l1.forward(hl1).select(output_buckets).screlu();
            let hl3 = l2.forward(hl2).select(output_buckets).screlu();
            l3.forward(hl3).select(output_buckets)
        });

    let no_clipping = AdamWParams { min_weight: -128.0, max_weight: 128.0, ..Default::default()};

    trainer.optimiser.set_params_for_weight("l2w", no_clipping);
    trainer.optimiser.set_params_for_weight("l2b", no_clipping);
    trainer.optimiser.set_params_for_weight("l3w", no_clipping);
    trainer.optimiser.set_params_for_weight("l3b", no_clipping);

    let wdl_scheduler = wdl::Sequence {
        first: wdl::ConstantWDL { value: 0.0 },
        second: wdl::ConstantWDL { value: 0.1 },
        first_scheduler_final_superbatch: STAGE1_SB,
    };

    let lr_scheduler = lr::Warmup {
        inner: lr::CosineDecayLR { initial_lr: s1_initial_lr, final_lr: s1_final_lr, final_superbatch: STAGE1_SB },
        warmup_batches: 800,
    };

    let schedule = TrainingSchedule {
        net_id: "moarlayers".to_string(),
        eval_scale: 362.0,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
            start_superbatch: 1,
            end_superbatch: STAGE1_SB,
        },
        wdl_scheduler,
        lr_scheduler,
        save_rate: 80,
    };

    let settings = LocalSettings { threads: 2, test_set: None, output_directory: "checkpoints", batch_queue_size: 64 };

    // loading from a SF binpack
    let dataloader = {
        let file_path = dataset_path;
        let buffer_size_mb = 4096;
        let threads = 4;
        fn filter(entry: &TrainingDataEntry) -> bool {
            entry.ply >= 20
                && !entry.pos.is_checked(entry.pos.side_to_move())
                && entry.score.unsigned_abs() <= 10000
                && entry.mv.mtype() == MoveType::Normal
                && entry.pos.piece_at(entry.mv.to()).piece_type() == PieceType::None
                && shouldkeep(entry.result, entry.score, &entry.pos)
        }
        SfBinpackLoader::new(file_path, buffer_size_mb, threads, filter)
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

    // trainer.load_from_checkpoint("checkpoints\\moarlayers-80");

    //trainer.save_to_checkpoint("checkpoints\\fixed-shit");

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
