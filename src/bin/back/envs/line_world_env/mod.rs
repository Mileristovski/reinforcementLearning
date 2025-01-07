use nalgebra::DVector;

use crate::back::envs::basic_env::Env;

pub struct LineEnv {
    s: DVector<i32>,
    a: DVector<i32>,
    r: DVector<i32>,
    t: Vec<usize>,
    p: Vec<Vec<Vec<Vec<f32>>>>,
    current_state: usize,
    current_score: f32,
}

impl LineEnv {
    pub fn new() -> Self {
        let s = DVector::from_vec(vec![0, 1, 2, 3, 4]);
        let a = DVector::from_vec(vec![0, 1]);
        let r = DVector::from_vec(vec![-1, 0, 1]);
        let t = vec![0, 4];
        let mut p = vec![vec![vec![vec![0.0f32; r.len()]; s.len()]; a.len()]; s.len()];

        for &s_p in &s {
            if s_p == 0 || s_p == 4 {
                continue;
            }
            for &action in &a {
                if action == 0 && s_p > 1 {
                    p[s_p as usize][action as usize][(s_p - 1) as usize][1] = 1.0;
                }
                if action == 1 && s_p < 3 {
                    p[s_p as usize][action as usize][(s_p + 1) as usize][1] = 1.0;
                }
            }
        }

        // Set probabilities
        p[1][0][0][0] = 1.0;
        p[3][1][4][2] = 1.0;
        let current_state = s.len() / 2;
        let current_score = 0.0;

        LineEnv { s, a, r, t, p, current_state, current_score }
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

    fn get_reward_vector(&self) -> Vec<f32> {
        vec![-1.0f32, 0.0, 1.0]
    }

    fn get_terminal_states(&self) -> Vec<usize> {
        self.t.clone()
    }

    fn get_reward(&self, _num: usize) -> f32 {
        self.current_score
    }

    fn get_action_spaces(&self) -> Vec<usize> {
        todo!()
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

    fn transition_probability(&self, state: usize, a: usize, s_p: usize, r_index: usize) -> f32 {
        self.p[state][a][s_p][r_index]
    }
}

