extern crate rand;

use crate::back::envs::basic_env::Env;
use rand::distributions::{WeightedIndex, Distribution, Uniform};
use kdam::tqdm;


pub fn off_policy_mc_control(
    env: &mut dyn Env,
    num_episodes: usize,
    gamma: f32,
) -> (Vec<usize>, Vec<Vec<f32>>) {
    let num_states = env.num_states();
    let num_actions = env.num_actions();

    // Initialize Q and C
    let mut q = vec![vec![0.0; num_actions]; num_states];
    let mut c = vec![vec![0.0; num_actions]; num_states];
    let mut pi = vec![0; num_states];

    for _ in tqdm!(0..num_episodes, position = 0) {
        let b = vec![vec![1.0 / num_actions as f32; num_actions]; num_states];
        let mut trajectory = Vec::new();
        let all_actions: Vec<usize> = (0..num_actions).collect();
        env.reset();

        // Generate an episode following behavior policy b
        while !env.is_game_over() {
            let s = env.state_id();
            let pi_s = &b[s];
            let dist = WeightedIndex::new(pi_s).unwrap();
            let mut rng = rand::thread_rng();
            let a: usize = all_actions[dist.sample(&mut rng)];

            let prev_score = env.score();
            env.step(a as i32);
            let r = env.score() - prev_score;
            trajectory.push((s, a, r));
        }

        let mut g = 0.0;
        let mut w = 1.0;

        // Backward pass through the trajectory
        for &(s_t, a_t, r_t) in trajectory.iter().rev() {
            g = gamma * g + r_t;

            c[s_t][a_t] += w;
            q[s_t][a_t] += w / c[s_t][a_t] * (g - q[s_t][a_t]);

            pi[s_t] = q[s_t].iter().enumerate().max_by(|x, y| x.1.partial_cmp(y.1).unwrap()).unwrap().0;

            if a_t != pi[s_t] {
                break;
            }
            w *= 1.0 / b[s_t].iter().sum::<f32>();
        }
    }
    (pi, q)
}

pub fn off_policy_mc_control_dynamic(
    env: &mut dyn Env,
    num_episodes: usize,
    gamma: f32,
    action_spaces: &[usize],
) -> (Vec<usize>, Vec<Vec<f32>>) {
    let num_states = env.num_states();

    // Initialize Q and C
    let mut q: Vec<Vec<f32>> = action_spaces
        .iter()
        .map(|&num_actions| vec![rand::random::<f32>(); num_actions])
        .collect();
    let mut c: Vec<Vec<f32>> = action_spaces
        .iter()
        .map(|&num_actions| vec![0.0; num_actions])
        .collect();
    let mut pi = vec![0; num_states]; // Each state maps to one action


    for _ in tqdm!(0..num_episodes, position = 0) {
        let b: Vec<Vec<f32>> = action_spaces
            .iter()
            .map(|&num_actions| {
                let mut actions: Vec<f32> = (0..num_actions).map(|_| rand::random::<f32>()).collect();
                let sum: f32 = actions.iter().sum();
                actions.iter_mut().for_each(|p| *p /= sum); // Normalize to sum to 1
                actions
            })
            .collect();

        let mut trajectory = Vec::new();
        env.reset();

        while !env.is_game_over() {
            let s = env.state_id();
            let pi_s = &b[s];
            let dist = WeightedIndex::new(pi_s).unwrap();
            let mut rng = rand::thread_rng();

            let available_actions = env.available_actions();

            let action_index = dist.sample(&mut rng);
            let a = available_actions[action_index] as usize; // Sample action for state `s`
            let prev_score = env.score();
            env.step(a as i32);
            let r = env.score() - prev_score;
            trajectory.push((s, action_index, r));
        }

        let mut g = 0.0;
        let mut w = 1.0;

        // Backward pass through the trajectory
        for &(s_t, a_t, r_t) in trajectory.iter().rev() {
            g = gamma * g + r_t;

            c[s_t][a_t] += w;
            q[s_t][a_t] += w / c[s_t][a_t] * (g - q[s_t][a_t]);

            pi[s_t] = q[s_t].iter().enumerate().max_by(|x, y| x.1.partial_cmp(y.1).unwrap()).unwrap().0;

            if a_t != pi[s_t] {
                break;
            }
            w *= 1.0 / b[s_t].iter().sum::<f32>();
        }
    }
    (pi, q)
}

pub fn off_policy_mc_control_secret(
    env: &mut dyn Env,
    num_episodes: usize,
    gamma: f32,
) -> (Vec<usize>, Vec<Vec<f32>>) {
    let num_states = env.num_states();
    let num_actions = env.num_actions();

    let mut q = vec![vec![0.0; num_actions]; num_states];
    let mut c = vec![vec![0.0; num_actions]; num_states];
    let mut pi = vec![0; num_states];
    for _ in tqdm!(0..num_episodes, position = 0) {
        let b = vec![vec![1.0 / num_actions as f32; num_actions]; num_states];

        let mut trajectory = Vec::new();
        env.reset();

        while !env.is_game_over() {
            let s = env.state_id();
            let pi_s = &b[s];
            let mut rng = rand::thread_rng();

            let available_actions = env.available_actions();

            let dist = Uniform::from(0..available_actions.len());
            let action_index = dist.sample(&mut rng);
            let a = available_actions[action_index] as usize; // Sample action for state `s`
            let prev_score = env.score();
            env.step(a as i32);
            let r = env.score() - prev_score;
            trajectory.push((s, action_index, r));
        }

        let mut g = 0.0;
        let mut w = 1.0;

        // Backward pass through the trajectory
        for &(s_t, a_t, r_t) in trajectory.iter().rev() {
            g = gamma * g + r_t;

            c[s_t][a_t] += w;
            q[s_t][a_t] += w / c[s_t][a_t] * (g - q[s_t][a_t]);

            pi[s_t] = q[s_t].iter().enumerate().max_by(|x, y| x.1.partial_cmp(y.1).unwrap()).unwrap().0;

            if a_t != pi[s_t] {
                break;
            }
            w *= 1.0 / b[s_t].iter().sum::<f32>();
        }
    }
    (pi, q)
}
