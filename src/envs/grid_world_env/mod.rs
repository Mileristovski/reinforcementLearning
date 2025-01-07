use nalgebra::DVector;

pub use crate::envs::basic_env::Env;

/// A simple Grid World environment.
/// The agent can move up, down, left, or right within a 2D grid.
///
/// - `s`: The states (optional usage – could be a flattened representation
///        of the grid if you want to store them).
/// - `a`: Possible actions (0=Up, 1=Down, 2=Left, 3=Right).
/// - `r`: Possible rewards per state or some placeholder vector for demonstration.
/// - `t`: Terminal states (stored as flattened indices).
/// - `rows`, `cols`: Dimensions of the grid.
/// - `current_state`: The agent's current position in the grid, also stored as a
///                    single integer via row-major flattening (row * cols + col).
/// - `current_score`: Accumulated score.
pub struct GridEnv {
    s: DVector<i32>,
    a: DVector<i32>,
    r: DVector<i32>,
    t: Vec<usize>,
    rows: usize,
    cols: usize,
    current_state: usize,
    current_score: f32,
}

impl GridEnv {
    /// Creates a new GridEnv with the given dimensions.
    /// `s_vec` can be used to represent states (optionally),
    /// `a_vec` for actions, `r_vec` for rewards, and `t_vec` for terminals.
    pub fn new(
        rows: usize,
        cols: usize,
        s_vec: Vec<i32>,
        a_vec: Vec<i32>,
        r_vec: Vec<i32>,
        t_vec: Vec<usize>,
    ) -> Self {
        let s = DVector::from_vec(s_vec);
        let a = DVector::from_vec(a_vec);
        let r = DVector::from_vec(r_vec);
        let t = t_vec;

        // For simplicity, start at middle cell if possible,
        // otherwise just start at 0.
        let grid_size = rows * cols;
        let starting_state = if grid_size > 0 { grid_size / 2 } else { 0 };

        GridEnv {s, a, r, t, rows, cols, current_state: starting_state, current_score: 0.0}
    }

    pub fn get_grid_state(&self) -> Vec<(usize, usize, bool)> {
        let mut grid = vec![];
        for row in 0..self.rows {
            for col in 0..self.cols {
                let index = self.rc_to_index(row, col);
                let is_current = index == self.current_state; // Only mark the current state
                grid.push((row, col, is_current));
            }
        }
        grid
    }

    /// Utility to convert a (row, col) pair into a single flattened index.
    fn rc_to_index(&self, row: usize, col: usize) -> usize {
        row * self.cols + col
    }

    /// Utility to convert a single flattened index into a (row, col) pair.
    fn index_to_rc(&self, idx: usize) -> (usize, usize) {
        let row = idx / self.cols;
        let col = idx % self.cols;
        (row, col)
    }
}

impl Env for GridEnv {
    fn num_states(&self) -> usize {
        self.s.len()
    }

    fn num_actions(&self) -> usize {
        self.a.len()
    }

    fn num_rewards(&self) -> usize {
        self.r.len()
    }

    fn reward(&self, _i: usize) -> f32 {
        // You could define a more interesting reward structure.
        // For now, just returning 0.0 as a placeholder.
        0.0
    }

    fn p(&self, _s: i32, _a: i32, _s_p: i32, _r_index: i32) -> f32 {
        // Probability of transition (s -> s') given action a, possibly for MDPs.
        // Not implemented here. Return 0.0 or panic.
        panic!("Not yet implemented");
    }

    fn state_id(&self) -> usize {
        self.current_state
    }

    fn reset(&mut self) {
        // Reset to the center of the grid or 0 if there's no space.
        let grid_size = self.rows * self.cols;
        self.current_state = if grid_size > 0 { grid_size / 2 } else { 0 };
        self.current_score = 0.0;
    }

    fn display(&self) {
        let (agent_row, agent_col) = self.index_to_rc(self.current_state);
        for row in 0..self.rows {
            for col in 0..self.cols {
                if row == agent_row && col == agent_col {
                    print!("X ");
                } else {
                    print!("_ ");
                }
            }
            println!();
        }
        println!();
    }

    fn is_forbidden(&self, _action: usize) -> bool {
        // If you'd like to forbid certain actions in certain states, implement logic here.
        // For this example, we do not forbid any action.
        false
    }

    fn is_game_over(&self) -> bool {
        // The game is over if the current state is a terminal state.
        self.t.contains(&self.current_state)
    }

    fn available_actions(&self) -> DVector<i32> {
        // If the game is over, no actions are available.
        // Otherwise, return all possible moves: Up(0), Down(1), Left(2), Right(3)
        if self.is_game_over() {
            DVector::zeros(0)
        } else {
            self.a.clone()
        }
    }

    fn step(&mut self, action: i32) {
        // Check if the action is valid.
        if !self.available_actions().iter().any(|&x| x == action) {
            panic!("Invalid action");
        }

        if self.is_game_over() {
            panic!("Trying to play when the game is over!");
        }

        let (mut row, mut col) = self.index_to_rc(self.current_state);

        match action {
            0 => { // Up
                if row > 0 {
                    row -= 1;
                }
            }
            1 => { // Down
                if row < self.rows - 1 {
                    row += 1;
                }
            }
            2 => { // Left
                if col > 0 {
                    col -= 1;
                }
            }
            3 => { // Right
                if col < self.cols - 1 {
                    col += 1;
                }
            }
            _ => panic!("Invalid action"),
        };

        self.current_state = self.rc_to_index(row, col);

        // For demonstration, let's say if top-left corner is 0, you lose,
        // bottom-right corner is a 'win' scenario.
        // This is purely arbitrary; feel free to replace with your own logic.
        if self.current_state == 0 {
            self.current_score = -1.0;
        } else if self.current_state == (self.rows * self.cols - 1) {
            self.current_score = 1.0;
        } else {
            // Otherwise, 0.0 or accumulate something else as you wish.
        }
    }

    fn score(&self) -> f32 {
        self.current_score
    }

    fn from_random_state() -> Self
    where
        Self: Sized,
    {
        // You might want to create a random grid or place the agent randomly.
        // For now, we panic to match the style of the example.
        panic!("Not yet implemented");
    }
}
