//! Small search algorithms with explicit Generate -> Evaluate -> Select phases.

mod cma_es;
pub(crate) mod common;
mod elitist;
mod es;
mod ga;
mod nsga2;
mod openai_es;

pub use cma_es::{CmaEs, CmaEsConfig};
pub use common::SearchError;
pub use elitist::{ElitistConfig, ElitistError, ElitistSearch, Objective};
pub use es::{Es, EsConfig, EsParent};
pub use ga::{Ga, GaConfig};
pub use nsga2::{Nsga2, Nsga2Config};
pub use openai_es::{OpenAiEs, OpenAiEsConfig, ScoreTransform};
