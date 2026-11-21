use super::{Activation, NnError, affine, check_len, dense_count, finite};

/// A tanh recurrent cell with state supplied explicitly by the caller.
///
/// For each hidden output, parameters are input weights, previous-hidden-state
/// weights, then one bias. The update is `h' = tanh(Wx + Uh + b)`.
/// This is one forward step; backpropagation through time is not implemented.
#[derive(Clone, Debug)]
pub struct RnnCell {
    inputs: usize,
    hidden: usize,
    parameters: Vec<f64>,
}

impl RnnCell {
    /// Number of input weights, recurrent weights, and biases.
    pub fn parameter_count(inputs: usize, hidden: usize) -> Result<usize, NnError> {
        if inputs == 0 || hidden == 0 {
            return Err(NnError::InvalidShape("RNN dimensions must be positive"));
        }
        let combined = inputs
            .checked_add(hidden)
            .ok_or(NnError::Overflow("RNN input width"))?;
        dense_count(combined, hidden)
    }

    /// Construct a cell after checking dimensions and finite parameters.
    pub fn new(inputs: usize, hidden: usize, parameters: Vec<f64>) -> Result<Self, NnError> {
        check_len(
            &parameters,
            Self::parameter_count(inputs, hidden)?,
            "RNN parameters",
        )?;
        Ok(Self {
            inputs,
            hidden,
            parameters,
        })
    }

    /// Number of input coordinates per step.
    pub fn input_size(&self) -> usize {
        self.inputs
    }

    /// Number of hidden-state coordinates.
    pub fn hidden_size(&self) -> usize {
        self.hidden
    }

    /// Flat parameters in hidden-neuron order.
    pub fn parameters(&self) -> &[f64] {
        &self.parameters
    }

    /// A fresh zero state, suitable for starting an independent sequence.
    pub fn zero_state(&self) -> Vec<f64> {
        vec![0.0; self.hidden]
    }

    /// Return the next state without modifying the cell or the previous state.
    pub fn step(&self, input: &[f64], previous: &[f64]) -> Result<Vec<f64>, NnError> {
        check_len(input, self.inputs, "RNN input")?;
        check_len(previous, self.hidden, "RNN previous state")?;
        let combined = input.iter().chain(previous).copied().collect::<Vec<_>>();
        Ok(affine(&combined, &self.parameters)?
            .into_iter()
            .map(f64::tanh)
            .collect())
    }
}

/// External hidden and cell state of an LSTM.
///
/// [`LstmCell::step`] checks both vectors on every call, including states built
/// or modified directly by users. Independent sequences require independent states.
#[derive(Clone, Debug, PartialEq)]
pub struct LstmState {
    /// Hidden output vector `h`.
    pub hidden: Vec<f64>,
    /// Memory vector `c`.
    pub cell: Vec<f64>,
}

impl LstmState {
    /// Create a zero state with a positive number of coordinates.
    pub fn zeros(hidden: usize) -> Result<Self, NnError> {
        if hidden == 0 {
            return Err(NnError::InvalidShape("LSTM state width must be positive"));
        }
        Ok(Self {
            hidden: vec![0.0; hidden],
            cell: vec![0.0; hidden],
        })
    }
}

/// A forward-only LSTM cell with input, forget, candidate, and output gates.
///
/// Parameter groups are ordered `i, f, g, o`. Within each group, each hidden
/// neuron stores input weights, previous-hidden weights, then its bias.
/// The candidate uses tanh; the other gates use sigmoid. State follows
/// `c' = f*c + i*g`, `h' = o*tanh(c')`. There are no peepholes or projections.
#[derive(Clone, Debug)]
pub struct LstmCell {
    inputs: usize,
    hidden: usize,
    parameters: Vec<f64>,
}

impl LstmCell {
    /// Number of parameters across all four gates.
    pub fn parameter_count(inputs: usize, hidden: usize) -> Result<usize, NnError> {
        if inputs == 0 || hidden == 0 {
            return Err(NnError::InvalidShape("LSTM dimensions must be positive"));
        }
        let combined = inputs
            .checked_add(hidden)
            .ok_or(NnError::Overflow("LSTM input width"))?;
        let outputs = hidden
            .checked_mul(4)
            .ok_or(NnError::Overflow("LSTM gate count"))?;
        dense_count(combined, outputs)
    }

    /// Construct a cell from finite parameters in documented gate order.
    pub fn new(inputs: usize, hidden: usize, parameters: Vec<f64>) -> Result<Self, NnError> {
        check_len(
            &parameters,
            Self::parameter_count(inputs, hidden)?,
            "LSTM parameters",
        )?;
        Ok(Self {
            inputs,
            hidden,
            parameters,
        })
    }

    /// Number of input coordinates per step.
    pub fn input_size(&self) -> usize {
        self.inputs
    }

    /// Number of hidden and cell coordinates.
    pub fn hidden_size(&self) -> usize {
        self.hidden
    }

    /// Flat parameters grouped by input, forget, candidate, and output gates.
    pub fn parameters(&self) -> &[f64] {
        &self.parameters
    }

    /// A fresh zero state for an independent sequence.
    pub fn zero_state(&self) -> LstmState {
        LstmState {
            hidden: vec![0.0; self.hidden],
            cell: vec![0.0; self.hidden],
        }
    }

    /// Compute a new state while leaving the prior state unchanged on all paths.
    pub fn step(&self, input: &[f64], previous: &LstmState) -> Result<LstmState, NnError> {
        check_len(input, self.inputs, "LSTM input")?;
        check_len(&previous.hidden, self.hidden, "LSTM hidden state")?;
        check_len(&previous.cell, self.hidden, "LSTM cell state")?;
        let combined = input
            .iter()
            .chain(&previous.hidden)
            .copied()
            .collect::<Vec<_>>();
        let gates = affine(&combined, &self.parameters)?;
        let mut next = self.zero_state();
        for index in 0..self.hidden {
            let input_gate = Activation::Sigmoid.apply(gates[index]);
            let forget_gate = Activation::Sigmoid.apply(gates[self.hidden + index]);
            let candidate = gates[2 * self.hidden + index].tanh();
            let output_gate = Activation::Sigmoid.apply(gates[3 * self.hidden + index]);
            next.cell[index] = finite(
                forget_gate * previous.cell[index] + input_gate * candidate,
                "LSTM cell update",
            )?;
            next.hidden[index] = output_gate * next.cell[index].tanh();
        }
        Ok(next)
    }
}
