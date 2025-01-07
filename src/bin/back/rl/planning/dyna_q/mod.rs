use kdam::tqdm;

use rand::seq::IteratorRandom;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use std::collections::HashMap;

use crate::back::envs::basic_env::Env;
use crate::back::services::math::{max, epsilon_greedy_action};

pub fn dyna_q(
    env: &mut dyn Env,
    max_episodes: usize,
    alpha: f32,
    epsilon: f32,
    gamma: f32,
    planning_steps: usize,
) -> (Vec<Vec<f32>>, HashMap<(usize, i32), (usize, f32)>) {
    let mut rng = Xoshiro256PlusPlus::from_entropy();

    // Initialize Q-table
    let mut q = Vec::new();
    for _ in 0..env.num_states() {
        let mut v = Vec::new();
        for _ in 0..env.num_actions() {
            v.push(0.0f32);
        }
        q.push(v);
    }

    // Initialize the model as a HashMap
    let mut model: HashMap<(usize, i32), (usize, f32)> = HashMap::new();

    for _ in tqdm!(0..max_episodes, position = 0) {
        env.reset();
        let mut s = env.state_id();
        let mut aa = env.available_actions();

        while !env.is_game_over() {
            // Step (b): Choose action A using epsilon-greedy
            let a = epsilon_greedy_action(aa.clone(), &q, s, epsilon, &mut rng);

            let prev_score = env.score();
            env.step(a);
            let r = env.score() - prev_score;
            let s_p = env.state_id();
            let aa_p = env.available_actions();

            // Step (d): Update Q(S, A)
            let max_q_s_p = if env.is_game_over() {
                0.0f32
            } else {
                max(&q[s_p])
            };
            q[s][a as usize] += alpha * (r + gamma * max_q_s_p - q[s][a as usize]);

            // Step (e): Update the model
            model.insert((s, a), (s_p, r));

            // Step (f): Perform planning
            for _ in 0..planning_steps {
                // Randomly sample a previously observed state-action pair
                let &(s_rand, a_rand) = model.keys().choose(&mut rng).unwrap();
                let &(s_rand_p, r_rand) = model.get(&(s_rand, a_rand)).unwrap();

                // Update Q(S, A) based on the simulated experience
                let max_q_rand_s_p = if env.is_game_over() {
                    0.0f32
                } else {
                    max(&q[s_rand_p])
                };
                q[s_rand][a_rand as usize] +=
                    alpha * (r_rand + gamma * max_q_rand_s_p - q[s_rand][a_rand as usize]);
            }

            // Move to the next state
            s = s_p;
            aa = aa_p;
        }
    }

    (q.clone(), model.clone())
}
