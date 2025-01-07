use kdam::tqdm;

use rand::prelude::SliceRandom;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::back::envs::basic_env::Env;
use crate::back::services::math::{max, epsilon_greedy_action};

pub fn q_learning(env: &mut dyn Env, max_episodes: usize, alpha: f32, epsilon: f32, gamma: f32) -> Vec<Vec<f32>> {
    let mut rng = Xoshiro256PlusPlus::from_entropy();

    let mut q = Vec::new();
    for _ in 0..env.num_states() {
        let mut v = Vec::new();
        for _ in 0..env.num_actions() {
            v.push(0.0f32);
        }
        q.push(v);
    }

    for _ in tqdm!(0..max_episodes, position = 0) {
        env.reset();
        let mut s = env.state_id();
        let mut aa = env.available_actions();

        while !env.is_game_over() {
            let a = epsilon_greedy_action(aa, &q, s, epsilon, &mut rng);

            let prev_score = env.score();
            env.step(a);
            let r = env.score() - prev_score;

            let s_p = env.state_id();
            let aa_p = env.available_actions();

            let max_q_s_p = if env.is_game_over() {
                0.0f32
            } else {
                max(&q[s_p])
            };

            q[s][a as usize] = q[s][a as usize] + alpha * (r + gamma * max_q_s_p - q[s][a as usize]);

            s = s_p;
            aa = aa_p;
        }
    }

    q.clone()
}

pub fn q_learning_dynamic(env: &mut dyn Env, max_episodes: usize, alpha: f32, epsilon: f32, gamma: f32, action_spaces: &[usize]) -> Vec<Vec<f32>> {
    let mut rng = Xoshiro256PlusPlus::from_entropy();

    let mut q: Vec<Vec<f32>> = action_spaces
        .iter()
        .map(|&num_actions| vec![0.0f32; num_actions])
        .collect();

    for _ in tqdm!(0..max_episodes, position = 0) {
        env.reset();
        let mut s = env.state_id();
        let mut aa = env.available_actions();

        while !env.is_game_over() {

            let rnd_number = rand::random::<f32>();
            let (a_index, a) = if rnd_number <= epsilon {
                let indexed_aa: Vec<(usize, &i32)> = aa.as_slice().iter().enumerate().collect();
                let &(index, &value) = indexed_aa.choose(&mut rng).unwrap();
                (index, value)
            } else {
                let mut best_a = 0;
                let mut best_a_score = f32::MIN;
                let mut best_index = 0;

                for (index, &a) in aa.iter().enumerate() {
                    let q_s_a = q[s][index];

                    if q_s_a >= best_a_score {
                        best_a = a;
                        best_a_score = q_s_a;
                        best_index = index;
                    }
                }
                (best_index, best_a)
            };

            let prev_score = env.score();
            env.step(a);
            let r = env.score() - prev_score;

            let s_p = env.state_id();
            let aa_p = env.available_actions();

            let max_q_s_p = if env.is_game_over() {
                0.0f32
            } else {
                max(&q[s_p])
            };

            q[s][a_index] = q[s][a_index] + alpha * (r + gamma * max_q_s_p - q[s][a_index as usize]);

            s = s_p;
            aa = aa_p;
        }
    }

    q.clone()
}
