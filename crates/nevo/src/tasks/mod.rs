//! Small executable environments for teaching neuroevolution.

mod target_reach;
mod xor;

pub use target_reach::{Rollout, State, TargetReach, TargetReachError};
pub use xor::Xor;
