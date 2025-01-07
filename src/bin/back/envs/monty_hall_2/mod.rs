use rand::Rng;
use nalgebra::DVector;
pub use crate::back::envs::basic_env::Env;

pub struct MontyHallLevel2Env {
    winning_door: usize,
    chosen_door: Option<usize>,
    revealed_doors: Vec<usize>,
    step: usize,
    reward: f32,
    action_spaces: Vec<usize>,
}

impl MontyHallLevel2Env {
    pub fn new() -> Self {
        let winning_door = rand::thread_rng().gen_range(0..5);
        let action_spaces = vec![5, 4, 3, 2];
        MontyHallLevel2Env {
            winning_door,
            chosen_door: None,
            revealed_doors: Vec::new(),
            step: 0,
            reward: 0.0,
            action_spaces
        }
    }

    fn reveal_non_winning_door(&mut self) {
        // Find a door to reveal that is not the chosen door and not the winning door
        let door_to_reveal = (0..5)
            .filter(|&door| Some(door) != self.chosen_door && door != self.winning_door && !self.revealed_doors.contains(&door))
            .next()
            .unwrap();

        self.revealed_doors.push(door_to_reveal);
    }
}

impl Env for MontyHallLevel2Env {
    fn num_states(&self) -> usize {
        5
    }

    fn num_actions(&self) -> usize {
        4 // Switch to another door or stay (4 options total, including staying with the current door)
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
        self.winning_door = rand::thread_rng().gen_range(0..5);
        self.chosen_door = None;
        self.revealed_doors.clear();
        self.step = 0;
        self.reward = 0.0;
    }

    fn display(&self) {
        println!("Step: {}", self.step);
        println!("Winning Door: {} (hidden)", self.winning_door);
        println!("Chosen Door: {:?}", self.chosen_door);
        println!("Revealed Doors: {:?}", self.revealed_doors);
        println!("Reward: {}", self.reward);
    }

    fn is_forbidden(&self, action: usize) -> bool {
        // The action must not be a revealed door or an invalid door index
        self.revealed_doors.contains(&action) || action >= 5
    }

    fn is_game_over(&self) -> bool {
        self.step >= 4 // The game ends after 4 steps
    }

    fn available_actions(&self) -> DVector<i32> {
        if self.is_game_over() {
            DVector::zeros(0)
        } else {
            let mut actions: Vec<i32> = (0..5)
                .filter(|&door| !self.revealed_doors.contains(&door))
                .map(|door| door as i32)
                .collect();

            // Move chosen door to the front if it exists
            if let Some(chosen) = self.chosen_door {
                if let Some(pos) = actions.iter().position(|&x| x == chosen as i32) {
                    actions.remove(pos);
                    actions.insert(0, chosen as i32);
                }
            }

            DVector::from_vec(actions)
        }
    }
    fn step(&mut self, action: i32) {
        if self.is_game_over() {
            panic!("Game is already over");
        }

        let chosen_door = action as usize;

        if self.is_forbidden(chosen_door) {
            panic!("Invalid action: Door {} is already revealed", chosen_door);
        }

        self.chosen_door = Some(chosen_door);

        if self.step < 3 {
            // Reveal a door in the first 3 steps
            self.reveal_non_winning_door();
        } else if self.step == 3 {
            // Final step: Determine if the chosen door is the winning door
            if self.chosen_door.unwrap() == self.winning_door {
                self.reward = 1.0;
            } else {
                self.reward = 0.0;
            }
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
