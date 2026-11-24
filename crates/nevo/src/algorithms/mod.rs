//! Small search algorithms with explicit Generate -> Evaluate -> Select phases.

pub(crate) mod batch;
mod cma_es;
pub(crate) mod common;
mod deep_ga;
mod differential_evolution;
mod elitist;
mod es;
mod ga;
mod neat;
mod nsga2;
mod openai_es;

pub use cma_es::{CmaEs, CmaEsConfig};
pub use common::SearchError;
pub use deep_ga::{DeepGa, DeepGaConfig};
pub use differential_evolution::{De, DeConfig};
pub use elitist::{ElitistConfig, ElitistError, ElitistSearch, Objective};
pub use es::{Es, EsConfig, EsParent, EsSelection, EsStepSize};
pub use ga::{Ga, GaConfig};
pub use neat::{Neat, NeatConfig};
pub use nsga2::{Nsga2, Nsga2Config};
pub use openai_es::{OpenAiEs, OpenAiEsConfig, ScoreTransform};
