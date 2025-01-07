extern crate rand;

use crate::back::envs::basic_env::Env;
use rand::Rng;
use std::collections::HashMap;

pub fn monte_carlo_es(
    env: &mut dyn Env,
    num_episodes: usize,
    gamma: f32,
) -> (Vec<usize>, Vec<Vec<f32>>) {
    let num_states = env.num_states();
    let num_actions = env.num_actions();

    let mut q = vec![vec![0.0; num_actions]; num_states];
    let mut returns: Vec<Vec<Vec<f32>>> = vec![vec![Vec::new(); num_actions]; num_states];
    let mut pi = vec![0; num_states];

    for _ in 0..num_episodes {
        env.reset();
        let mut rng = rand::thread_rng();
        let s0 = rng.gen_range(0..num_states);
        let a0 = rng.gen_range(0..num_actions);

        // env.set_state(s0);
        env.step(a0 as i32);

        let mut trajectory = vec![(s0, a0, 0.0)];


        while !env.is_game_over() {
            let s = env.state_id();
            let a = pi[s];
            let prev_score = env.score();
            env.step(a as i32);
            let r = env.score() - prev_score;
            trajectory.push((s, a, r));
        }

        let mut g = 0.0;
        let mut visited = HashMap::new();

        for (t, &(s_t, a_t, r_t)) in trajectory.iter().rev().enumerate() {
            g = gamma * g + r_t;

            if !visited.contains_key(&(s_t, a_t)) {
                visited.insert((s_t, a_t), true);
                returns[s_t][a_t].push(g);
                q[s_t][a_t] = returns[s_t][a_t].iter().sum::<f32>() / returns[s_t][a_t].len() as f32;

                pi[s_t] = q[s_t]
                    .iter()
                    .enumerate()
                    .max_by(|(_, q1), (_, q2)| q1.partial_cmp(q2).unwrap())
                    .unwrap()
                    .0;
            }
        }
    }

    (pi, q)
}
