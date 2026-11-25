//! Small executable environments for teaching neuroevolution.

mod classification;
mod control;
mod target_reach;
mod xor;

pub use classification::{
    ClassificationData, ClassificationError, ClassificationMetrics, DataSplit, LabeledSample,
    MlpClassifier, NeatClassifier, NeatXor,
};
pub use control::{
    CartPoleTask, ControlTaskError, DoublePoleTask, NeatDoublePoleTask, SpringWalkerTask,
};
pub use target_reach::{Rollout, State, TargetReach, TargetReachError};
pub use xor::Xor;
