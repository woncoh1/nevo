//! Small search algorithms with explicit Generate -> Evaluate -> Select phases.

mod elitist;

pub use elitist::{ElitistConfig, ElitistError, ElitistSearch, Objective};
