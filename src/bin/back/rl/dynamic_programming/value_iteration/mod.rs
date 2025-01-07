use rand::Rng;
use crate::back::envs::basic_env::Env;

pub fn value_iteration(
    s: &Vec<usize>,  // States (as indices)
    a: &Vec<usize>,  // Actions
    r: &Vec<f32>,    // Rewards
    env: &mut dyn Env, // Transition probabilities
    gamma: f32,
    theta: f32,
) -> (Vec<f32>, Vec<usize>) {
    // Initialize value function V(s) arbitrarily (e.g., all zeros)
    let mut rng = rand::thread_rng();
    let mut v: Vec<f32> = (0..s.len()).map(|_| rng.gen()).collect();

    loop {
        let mut delta: f32 = 0.0;

        for &state in s {
            let v_old = v[state];
            let mut best_value = -f32::INFINITY;

            for &action in a {
                let mut total = 0.0;

                for &next_state in s {
                    for r_id in 0..r.len() {
                        let reward = r[r_id];
                        total += env.transition_probability(state, action, next_state, r_id) * (reward + gamma * v[next_state]);
                    }
                }

                if total > best_value {
                    best_value = total;
                }
            }

            v[state] = best_value;
            delta = delta.max((v_old - v[state]).abs());
        }

        if delta < theta {
            break;
        }
    }

    // Extract the deterministic policy π(s) from the optimal value function V(s)
    let mut pi = vec![0; s.len()];

    for &state in s {
        let mut best_a = 0;
        let mut best_a_score = -f32::INFINITY;

        for &action in a {
            let mut total = 0.0;

            for &next_state in s {
                for r_id in 0..r.len() {
                    let reward = r[r_id];
                    total += env.transition_probability(state, action, next_state, r_id) * (reward + gamma * v[next_state]);
                }
            }

            if total > best_a_score {
                best_a = action;
                best_a_score = total;
            }
        }

        pi[state] = best_a;
    }
    (v, pi)
}
