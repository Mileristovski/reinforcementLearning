extern crate rand;

use crate::envs::Env;
use rand::distributions::{WeightedIndex, Distribution};
use kdam::tqdm;
use std::collections::HashMap;

pub fn on_policy_first_visit_monte_carlo_control(
    env: &mut dyn Env,
    num_episodes: usize,
    epsilon: f32,
    gamma: f32,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let num_states = env.num_states();
    let num_actions = env.num_actions();

    let mut pi = vec![vec![1.0 / num_actions as f32; num_actions]; num_states];
    let mut q = vec![vec![rand::random::<f32>(); num_actions]; num_states];
    let mut returns: Vec<Vec<Vec<f32>>> = vec![vec![Vec::new(); num_actions]; num_states];
    let all_actions: Vec<usize> = (0..num_actions).collect();

    for _ in tqdm!(0..num_episodes, position = 0) {
        let mut trajectory = Vec::new();
        env.reset();
        while !env.is_game_over() {
            let s = env.state_id();
            let pi_s = &pi[s];
            let dist = WeightedIndex::new(pi_s).unwrap();
            let mut rng = rand::thread_rng();
            let a: usize = all_actions[dist.sample(&mut rng)];

            let prev_score = env.score();
            env.step(a as i32);
            let r = env.score() - prev_score;
            trajectory.push((s, a, r));
        }

        let mut g = 0.0;
        let mut visited = HashMap::new();

        for (_, (s_t, a_t, r_t)) in trajectory.iter().rev().enumerate() {
            g = gamma * g + *r_t;
            if !visited.contains_key(&(*s_t, *a_t)) {
                visited.insert((*s_t, *a_t), true);
                returns[*s_t][*a_t].push(g);
                q[*s_t][*a_t] = returns[*s_t][*a_t].iter().sum::<f32>() / returns[*s_t][*a_t].len() as f32;
                let best_a = q[*s_t].iter().enumerate().max_by(|x, y| x.1.partial_cmp(y.1).unwrap()).unwrap().0;

                for a in 0..num_actions {
                    if a == best_a {
                        pi[*s_t][a] = 1.0 - epsilon + epsilon / num_actions as f32;
                    } else {
                        pi[*s_t][a] = epsilon / num_actions as f32;
                    }
                }
            }
        }
    }
    (pi, q)
}
