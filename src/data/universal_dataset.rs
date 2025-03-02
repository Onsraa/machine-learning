use serde::{Serialize, Deserialize};

/// Indicates the problem type: regression or classification.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum TaskType {
    Regression,
    Classification,
}