use bullet_lib::{
    default::inputs::ChessBucketsMirroredFactorised, nn::{
        optimiser::{AdamWOptimiser, AdamWParams},
        ExecutionContext, Graph, NetworkBuilder, Node, Shape,
    }, trainer::{default::{inputs, loader, outputs, Trainer}, save::{Layout, QuantTarget, SavedFormat}, schedule::{lr, wdl, TrainingSchedule, TrainingSteps}, settings::LocalSettings, NetworkTrainer}
};
use bulletformat::ChessBoard;
use sfbinpack::chess::{r#move::MoveType, piecetype::PieceType};

const HL: usize = 1536;
const L2: usize = 16;
const L3: usize = 32;

type Input = ChessBucketsMirroredFactorised;

#[derive(Clone, Copy, Default)]
pub struct CJBucket;
impl bullet_lib::default::outputs::OutputBuckets<bulletformat::ChessBoard> for CJBucket {
    const BUCKETS: usize = 8;

    fn bucket(&self, pos: &bulletformat::ChessBoard) -> u8 {
        let pc_count = pos.occ().count_ones();
        ((63 - pc_count) * (32 - pc_count) / 225).min(7) as u8
    }
}

type Output = CJBucket;

fn main() {
    #[rustfmt::skip]
    let inputs = Input::new([
         0,  1,  2,  3,
         4,  5,  6,  7,
         8,  9, 10, 11,
         8,  9, 10, 11,
        12, 12, 13, 13,
        12, 12, 13, 13,
        14, 14, 15, 15,
        14, 14, 15, 15,
    ]);
    let output_buckets = Output::default();

    let num_inputs = inputs::SparseInputType::num_inputs(&inputs);
    let max_active = inputs::SparseInputType::max_active(&inputs);
    let num_buckets = <Output as outputs::OutputBuckets<ChessBoard>>::BUCKETS;

    let (graph, output_node) = build_network(num_inputs, max_active, num_buckets, HL);

    let optimiser = AdamWParams {
        decay: 0.01,
        beta1: 0.9,
        beta2: 0.999,
        ..AdamWParams::default()
    };

    let mut trainer = Trainer::<AdamWOptimiser, _, _>::new(
        graph,
        output_node,
        optimiser,
        inputs,
        output_buckets,
        vec![
            SavedFormat::new("l0w", QuantTarget::Float, Layout::Normal),
            SavedFormat::new("l0b", QuantTarget::Float, Layout::Normal),
            SavedFormat::new("l1w", QuantTarget::Float, Layout::Normal),
            SavedFormat::new("l1b", QuantTarget::Float, Layout::Normal),
            SavedFormat::new("l2w", QuantTarget::Float, Layout::Normal),
            SavedFormat::new("l2b", QuantTarget::Float, Layout::Normal),
            SavedFormat::new("l3w", QuantTarget::Float, Layout::Normal),
            SavedFormat::new("l3b", QuantTarget::Float, Layout::Normal),
        ],
        false,
    );

    let no_clipping = AdamWParams { min_weight: -128.0, max_weight: 128.0, ..optimiser };

    trainer.optimiser_mut().set_params_for_weight("l2w", no_clipping);
    trainer.optimiser_mut().set_params_for_weight("l2b", no_clipping);
    trainer.optimiser_mut().set_params_for_weight("l3w", no_clipping);
    trainer.optimiser_mut().set_params_for_weight("l3b", no_clipping);

    let initial_lr = 0.001;
    let final_lr = 0.001 * 0.3 * 0.3 * 0.3 * 0.3 * 0.3 * 0.3 * 0.3;
    let sbs = 800;

    let schedule = TrainingSchedule {
        net_id: "falke".into(),
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
            start_superbatch: 1,
            end_superbatch: sbs,
        },
        eval_scale: 400.0,
        wdl_scheduler: wdl::ConstantWDL { value: 0.0 },
        lr_scheduler: lr::Warmup {
            inner: lr::LinearDecayLR { initial_lr, final_lr, final_superbatch: sbs },
            warmup_batches: 800,
        },
        save_rate: 80,
    };

    let settings = LocalSettings { threads: 4, test_set: None, output_directory: "checkpoints", batch_queue_size: 512 };

    let data_loader = {
        let file_path = "data/master.binpack";
        let buffer_size_mb = 4096;
        let threads = 4;
        fn filter(entry: &sfbinpack::TrainingDataEntry) -> bool {
                entry.ply >= 20
                && !entry.pos.is_checked(entry.pos.side_to_move())
                && entry.score.unsigned_abs() <= 10000
                && entry.mv.mtype() == MoveType::Normal
                && entry.pos.piece_at(entry.mv.to()).piece_type() == PieceType::None
        }
        loader::SfBinpackLoader::new(file_path, buffer_size_mb, threads, filter)
    };

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

fn build_network(num_inputs: usize, max_active: usize, output_buckets: usize, hl: usize) -> (Graph, Node) {
    let builder = NetworkBuilder::default();

    // inputs
    let stm = builder.new_sparse_input("stm", Shape::new(num_inputs, 1), max_active);
    let nstm = builder.new_sparse_input("nstm", Shape::new(num_inputs, 1), max_active);
    let targets = builder.new_dense_input("targets", Shape::new(1, 1));
    let buckets = builder.new_sparse_input("buckets", Shape::new(output_buckets, 1), 1);

    // trainable weights
    let l0 = builder.new_affine("l0", num_inputs, hl);
    let l1 = builder.new_affine("l1", hl, output_buckets * L2);
    let l2 = builder.new_affine("l2", L2, output_buckets * L3);
    let l3 = builder.new_affine("l3", L3, output_buckets);

    // 32 + 32 due to feature factoriser
    l0.init_with_effective_input_size(64);

    // inference
    let stm_subnet = l0.forward(stm).crelu().pairwise_mul();
    let ntm_subnet = l0.forward(nstm).crelu().pairwise_mul();
    let out = stm_subnet.concat(ntm_subnet);
    let out = l1.forward(out).select(buckets).screlu();
    let out = l2.forward(out).select(buckets).screlu();
    let out = l3.forward(out).select(buckets);

    let pred = out.sigmoid();
    pred.power_error(targets, 2.5);

    // graph, output node
    let output_node = out.node();
    (builder.build(ExecutionContext::default()), output_node)
}
