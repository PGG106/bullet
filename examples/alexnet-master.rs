    
use bullet_lib::{
    nn::{optimiser, Activation},
    trainer::{
        default::{
            formats::sfbinpack::{
                chess::{piecetype::PieceType, r#move::MoveType},
                TrainingDataEntry,
            },
            inputs, loader, outputs, Loss, TrainerBuilder,
        },
        schedule::{lr, wdl, TrainingSchedule, TrainingSteps},
        settings::LocalSettings,
    },
};

macro_rules! net_id {
    () => {
        "master-net"
    };
}

#[derive(Clone, Copy, Default)]
pub struct CJBucket;
impl bullet_lib::default::outputs::OutputBuckets<bulletformat::ChessBoard> for CJBucket {
    const BUCKETS: usize = 8;

    fn bucket(&self, pos: &bulletformat::ChessBoard) -> u8 {
        let pc_count = pos.occ().count_ones();
        ((63 - pc_count) * (32 - pc_count) / 225).min(7) as u8
    }
}

const NET_ID: &str = net_id!();

fn main() {
    #[rustfmt::skip]
    let mut trainer = TrainerBuilder::default()
        .quantisations(&[362, 64])
        .optimiser(optimiser::AdamW)
        .loss_fn(Loss::SigmoidMPE(2.5))
        .input(inputs::ChessBucketsMirroredFactorised::new([
            0,  1,  2,  3,
            4,  5,  6,  7,
            8,  9, 10, 11,
            8,  9, 10, 11,
            12, 12, 13, 13,
            12, 12, 13, 13,
            14, 14, 15, 15,
            14, 14, 15, 15
            ]))
        .output_buckets(CJBucket)
        .feature_transformer(1536)
        .activate(Activation::SCReLU)
        .add_layer(1)
        .build();

    let schedule = TrainingSchedule {
        net_id: NET_ID.to_string(),
        eval_scale: 400.0,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
            start_superbatch: 1,
            end_superbatch: 800,
        },
        wdl_scheduler: wdl::ConstantWDL { value: 0.0 },
        lr_scheduler: lr::StepLR { start: 0.001, gamma: 0.1, step: 160 },
        save_rate: 80,
    };

    trainer.set_optimiser_params(optimiser::AdamWParams {
            decay: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            min_weight: -1.414,
            max_weight: 1.414,
        });

    let settings = LocalSettings { threads: 4, test_set: None, output_directory: "checkpoints", batch_queue_size: 512 };

        // loading from a SF binpack
    let data_loader = {
        let file_path = "data/master.binpack";
        let buffer_size_mb = 4096;
        let threads = 4;
        fn filter(entry: &TrainingDataEntry) -> bool {
                entry.ply >= 20
                && !entry.pos.is_checked(entry.pos.side_to_move())
                && entry.score.unsigned_abs() <= 10000
                && entry.mv.mtype() == MoveType::Normal
                && entry.pos.piece_at(entry.mv.to).piece_type() == PieceType::None
        }
        loader::SfBinpackLoader::new(file_path, buffer_size_mb, threads, filter)
    };

  /*
      let data_loader = {
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
    //let data_loader = loader::DirectSequentialDataLoader::new(&["data/baseline.data"]);

    // trainer.load_from_checkpoint("checkpoints\\master-net-buckets-7-640");

    //trainer.save_to_checkpoint("checkpoints\\fixed-shit");

    trainer.run(&schedule, &settings, &data_loader);

    for fen in [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    ] {
        let eval = trainer.eval(fen);
        println!("FEN: {fen}");
        println!("EVAL: {}", 400.0 * eval);
    }
}
