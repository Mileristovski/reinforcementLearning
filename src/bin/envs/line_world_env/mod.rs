use nalgebra::{DVector};

pub mod env;
pub use env::Env;

pub struct LineEnv {
    s: DVector<i32>,
    a: DVector<i32>,
    r: DVector<i32>,
    t: Vec<usize>,
    current_state: usize,
    current_score: f32,
}

impl LineEnv {
    pub fn new(s: Vec<i32>, a: Vec<i32>, r: Vec<i32>, t: Vec<usize>,) -> Self {
        let s = DVector::from_vec(s);
        let a = DVector::from_vec(a);
        let r = DVector::from_vec(r);
        let t = t;

        let current_state = s.len() / 2;
        let current_score = 0.0;

        LineEnv { s, a, r, t, current_state, current_score }
    }
}

impl Env for LineEnv {
    fn num_states(&self) -> usize  {
        self.s.len()
    }

    fn num_actions(&self) -> usize  {
        self.a.len()
    }

    fn num_rewards(&self) -> usize  {
        self.r.len()
    }

    fn reward(&self, _i: usize) -> f32 {
        panic!("Not yet implemented")
    }

    fn p(&self, _s: i32, _a: i32, _s_p: i32, _r_index: i32) -> f32 {
        panic!("Not yet implemented")
    }

    fn state_id(&self) -> usize  {
        self.current_state
    }

    fn reset(&mut self) -> () {
        self.current_state = self.s.len() / 2;
        self.current_score = 0.0;
    }

    fn display(&self) {
        for s in 0..self.s.len() {
            if s == self.current_state {
                print!("X");
            } else {
                print!("_");
            }
        }
        println!();
    }

    fn is_forbidden(&self, _action: usize) -> bool {
        panic!("Not yet implemented")
    }

    fn is_game_over(&self) -> bool {
        self.t.contains(&self.current_state)
    }

    fn available_actions(&self) -> DVector<i32>  {
        if self.is_game_over() {
            DVector::zeros(0)
        } else {
            self.a.clone()
        }
    }

    fn step(&mut self, action: i32) -> () {
        if !self.available_actions().iter().any(|&x| x == action) {
            panic!("Invalid action");
        }

        if self.is_game_over() {
            panic!("Trying to play when game is over!")
        }

        if action == 0 {
            self.current_state -= 1
        } else if action == 1 {
            self.current_state += 1
        } else {
            panic!("Invalid action");
        }

        if self.current_state == 0 {
            self.current_score = -1.0
        } else if self.current_state == self.s.len() - 1 {
            self.current_score = 1.0
        }
    }

    fn score(&self) -> f32 {
        self.current_score
    }

    fn from_random_state() -> Self
    where
        Self: Sized,
    {
        panic!("Not yet implemented");
    }
}

