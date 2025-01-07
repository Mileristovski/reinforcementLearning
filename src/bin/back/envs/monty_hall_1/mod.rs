use rand::Rng;
use nalgebra::DVector;

pub use crate::back::envs::basic_env::Env;

pub struct MontyHallEnv {
    winning_door: usize,
    chosen_door: Option<usize>,
    remaining_door: Option<usize>,
    step: usize,
    reward: f32,
    action_spaces: Vec<usize>,
}

impl MontyHallEnv {
    pub fn new() -> Self {
        let winning_door = rand::thread_rng().gen_range(0..3);
        let action_spaces = vec![3, 2];
        MontyHallEnv {
            winning_door,
            chosen_door: None,
            remaining_door: None,
            step: 0,
            reward: 0.0,
            action_spaces
        }
    }

    fn reveal_remaining_door(&mut self) {
        if let Some(chosen) = self.chosen_door {
            self.remaining_door = Some(
                (0..3)
                    .filter(|&door| door != chosen && door != self.winning_door)
                    .next()
                    .unwrap(),
            );
        }
    }
}

impl Env for MontyHallEnv {
    fn num_states(&self) -> usize {
        2
    }

    fn num_actions(&self) -> usize {
        2 // Keep or Switch
    }

    fn num_rewards(&self) -> usize {
        1 // Single reward at the end
    }

    fn get_reward_vector(&self) -> Vec<f32> {
        panic!("Not yet implemented");
    }

    fn get_terminal_states(&self) -> Vec<usize> {
        todo!()
    }

    fn get_reward(&self, _num: usize) -> f32 {
        self.reward
    }

    fn get_action_spaces(&self) -> Vec<usize> {
        self.action_spaces.clone()
    }

    fn p(&self, _s: i32, _a: i32, _s_p: i32, _r_index: i32) -> f32 {
        panic!("Not yet implemented");
    }

    fn state_id(&self) -> usize {
        self.step
    }

    fn reset(&mut self) {
        self.winning_door = rand::thread_rng().gen_range(0..3);
        self.chosen_door = None;
        self.remaining_door = None;
        self.step = 0;
        self.reward = 0.0;
    }

    fn display(&self) {
        println!("Step: {}", self.step);
        println!("Winning Door: {} (hidden)", self.winning_door);
        println!("Chosen Door: {:?}", self.chosen_door);
        println!("Remaining Door: {:?}", self.remaining_door);
        println!("Reward: {}", self.reward);
    }

    fn is_forbidden(&self, action: usize) -> bool {
        if self.step == 0 && action > 2 { // First step: choose a door (0, 1, or 2)
            return true;
        }
        if self.step == 1 && action > 1 { // Second step: keep (0) or switch (1)
            return true;
        }
        false
    }

    fn is_game_over(&self) -> bool {
        self.step >= 2
    }

    fn available_actions(&self) -> DVector<i32> {
        match self.step {
            0 => DVector::from_vec(vec![0, 1, 2]), // Choose a door
            1 => DVector::from_vec(vec![0, 1]),   // Keep or Switch
            _ => DVector::zeros(0),              // No actions available
        }
    }

    fn step(&mut self, action: i32) {
        if self.is_game_over() {
            panic!("Game is already over");
        }

        match self.step {
            0 => {
                // First step: Choose a door
                self.chosen_door = Some(action as usize);
                self.reveal_remaining_door();
            }
            1 => {
                // Second step: Keep or Switch
                if action == 1 {
                    // Switch to the remaining door
                    self.chosen_door = Some((0..3).find(|&door| door != self.chosen_door.unwrap() && door != self.remaining_door.unwrap()).unwrap());
                }
                // Check if the final choice is the winning door
                if self.chosen_door.unwrap() == self.winning_door {
                    self.reward = 1.0;
                } else {
                    self.reward = 0.0;
                }
            }
            _ => panic!("Invalid step"),
        }

        self.step += 1;
    }

    fn score(&self) -> f32 {
        self.reward
    }

    fn from_random_state() -> Self
    where
        Self: Sized,
    {
        panic!("Not yet implemented");
    }

    fn transition_probability(&self, s: usize, a: usize, s_p: usize, r_index: usize) -> f32 {
        panic!("Not yet implemented");
    }
}
