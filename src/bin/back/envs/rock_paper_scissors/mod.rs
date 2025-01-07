use nalgebra::DVector;

pub use crate::back::envs::basic_env::Env;

pub struct RockPaperScissorsEnv {
    rounds: usize,
    current_round: usize,
    agent_choices: Vec<usize>,
    opponent_choices: Vec<usize>,
    rewards: Vec<i32>,
    total_score: i32
}

impl RockPaperScissorsEnv {
    pub fn new() -> Self {
        RockPaperScissorsEnv {
            rounds: 2,
            current_round: 0,
            agent_choices: Vec::new(),
            opponent_choices: Vec::new(),
            rewards: vec![0; 2],
            total_score: 0,
        }
    }

    fn calculate_reward(agent: usize, opponent: usize) -> i32 {
        match (agent, opponent) {
            (0, 2) | (1, 0) | (2, 1) => 1,  // Win: Rock > Scissors, Paper > Rock, Scissors > Paper
            (0, 1) | (1, 2) | (2, 0) => -1, // Loss: Rock < Paper, Paper < Scissors, Scissors < Rock
            _ => 0,                         // Draw
        }
    }

    fn generate_opponent_choice(&self) -> usize {
        if self.current_round == 0 {
            rand::random::<usize>() % 3 // Random choice for round 1
        } else {
            self.agent_choices[0] // Mimic agent's choice from round 1 in round 2
        }
    }
}

impl Env for RockPaperScissorsEnv {
    fn num_states(&self) -> usize {
        3 // Rock, Paper, Scissors
    }

    fn num_actions(&self) -> usize {
        3 // Rock, Paper, Scissors
    }

    fn num_rewards(&self) -> usize {
        2 // Two rounds
    }

    fn get_reward_vector(&self) -> Vec<f32> {
        panic!("Not yet implemented");
    }

    fn get_terminal_states(&self) -> Vec<usize> {
        panic!("Not yet implemented")
    }

    fn get_reward(&self, num: usize) -> f32 {
        self.rewards[num] as f32
    }

    fn get_action_spaces(&self) -> Vec<usize> {
        panic!("Not yet implemented")
    }

    fn p(&self, _s: i32, _a: i32, _s_p: i32, _r_index: i32) -> f32 {
        panic!("Not yet implemented")
    }

    fn state_id(&self) -> usize {
        self.current_round
    }

    fn reset(&mut self) {
        self.current_round = 0;
        self.agent_choices.clear();
        self.opponent_choices.clear();
        self.rewards = vec![0; 2];
        self.total_score = 0;
    }

    fn display(&self) {
        for round in 0..self.current_round {
            let agent = match self.agent_choices[round] {
                0 => "Rock",
                1 => "Paper",
                2 => "Scissors",
                _ => "Invalid",
            };
            let opponent = match self.opponent_choices[round] {
                0 => "Rock",
                1 => "Paper",
                2 => "Scissors",
                _ => "Invalid",
            };
            println!("Round {}: Agent -> {}, Opponent -> {}, Reward: {}", round + 1, agent, opponent, self.rewards[round]);
        }
        println!("Total Score: {}", self.total_score);
    }

    fn is_forbidden(&self, _action: usize) -> bool {
        false
    }

    fn is_game_over(&self) -> bool {
        self.current_round >= self.rounds
    }

    fn available_actions(&self) -> DVector<i32> {
        if self.is_game_over() {
            DVector::zeros(0)
        } else {
            DVector::from_vec(vec![0, 1, 2]) // Rock, Paper, Scissors
        }
    }

    fn step(&mut self, action: i32) {
        if self.is_game_over() {
            panic!("Game is already over");
        }

        let agent_choice = action as usize;
        let opponent_choice = self.generate_opponent_choice();

        self.agent_choices.push(agent_choice);
        self.opponent_choices.push(opponent_choice);

        let reward = Self::calculate_reward(agent_choice, opponent_choice);
        self.rewards[self.current_round] = reward;
        self.total_score += reward;

        self.current_round += 1;
    }

    fn score(&self) -> f32 {
        self.total_score as f32
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
