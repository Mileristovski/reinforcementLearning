use crate::back::envs::basic_env::Env;
use rand::Rng;

fn iterative_policy_evaluation(
    pi: &Vec<usize>,
    s: &Vec<usize>,
    r: &Vec<f32>,
    t: &Vec<usize>,
    env: &mut dyn Env,
    gamma: f32,
    theta: f32,
) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    let mut v: Vec<f32> = (0..s.len()).map(|_| rng.gen()).collect();
    for &T in t {
        v[T] = 0.0;
    }
    loop {
        let mut delta: f32 = 0.0;

        for &state in s {
            let v_old = v[state];
            let action = pi[state];
            let mut total = 0.0;

            for &next_state in s {
                for r_id in 0..r.len() {
                    let reward = r[r_id];
                    total += env.transition_probability(state, action, next_state, r_id) * (reward + gamma * v[next_state]);
                }
            }
            v[state] = total;
            delta = delta.max((v_old - v[state]).abs());
        }

        if delta < theta {
            break;
        }
    }
    v
}

pub fn policy_iteration(
    s: &Vec<usize>,
    a: &Vec<usize>,
    r: &Vec<f32>,
    t: &Vec<usize>,
    env: &mut dyn Env,
    gamma: f32,
    theta: f32,
) -> (Vec<usize>, Vec<f32>) {
    // Initialization
    let mut rng = rand::thread_rng();
    let mut v: Vec<f32> = (0..s.len()).map(|_| rng.gen()).collect();

    // Set terminal states to 0.0
    for &T in t {
        v[T] = 0.0;
    }
    let mut pi: Vec<usize> = vec![0; s.len()];
    loop {
        // Policy Evaluation
        v = iterative_policy_evaluation(&pi, &s, &r, &t, env, gamma, theta);

        // Policy Improvement
        let mut stable_policy = true;

        for &state in s {
            let old_action = pi[state];
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
            if best_a != old_action {
                stable_policy = false;
            }
        }

        if stable_policy {
            break;
        }
    }
    (pi, v)
}
