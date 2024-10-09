use rand::Rng;

/**
fn iterative_policy_evaluation(
    pi: &Vec<usize>,
    s: &Vec<usize>,
    a: &Vec<usize>,
    r: &Vec<f32>,
    p: &Vec<Vec<Vec<Vec<f32>>>>,
    t: &Vec<usize>,
    gamma: f32,
    theta: f32
) -> Vec<f32> {
    let mut v = vec![0.0_f32; s.len()];  // Initialize value function V(s) randomly
    for &term_state in t { // Set terminal state values to 0
        v[term_state] = 0.0;
    }

    loop {
        let mut delta: f32 = 0.0;

        for &state in s {
            let v_old = v[state];
            let mut total = 0.0;
            for &action in a {
                let mut action_value = 0.0;
                for &next_state in s {
                    for r_id in 0..r.len() {
                        action_value += p[state][action][next_state][r_id]
                            * (r[r_id] + gamma * v[next_state]);
                    }
                }
                total += pi[state][action] * action_value;
            }
            v[state] = total;
            delta = delta.max((v_old - v[state]).abs());
        }
        if delta <= theta {
            break;
        }
    }
    v
}
*/

fn iterative_policy_evaluation(
    pi: &Vec<usize>, // Policy (as indices)
    s: &Vec<usize>,  // States (as indices)
    r: &Vec<f32>,    // Rewards
    p: &Vec<Vec<Vec<Vec<f32>>>>, // Transition probabilities
    t: &Vec<usize>,  // Terminal states
    gamma: f32,
    theta: f32,
) -> Vec<f32> {
    let mut v = vec![0.0_f32; s.len()];  // Initialize value function V(s) randomly
    for &term_state in t { // Set terminal state values to 0
        v[term_state] = 0.0;
    }

    loop {
        let mut delta: f32 = 0.0;

        for &state in s {
            let v_old = v[state];
            let action = pi[state]; // Get the action from policy
            let mut total = 0.0;

            for &next_state in s {
                for r_id in 0..r.len() {
                    let reward = r[r_id];
                    total += p[state][action][next_state][r_id] * (reward + gamma * v[next_state]);
                }
            }
            v[state] = total;
            delta = delta.max((v_old - v[state]).abs());
        }
        // Check for convergence
        if delta <= theta {
            break;
        }
    }
    v
}

pub fn policy_iteration(
    s: &Vec<usize>,  // States (as indices)
    a: &Vec<usize>,  // Actions
    r: &Vec<f32>,    // Rewards
    p: &Vec<Vec<Vec<Vec<f32>>>>, // Transition probabilities
    t: &Vec<usize>,  // Terminal states
    gamma: f32,
    theta: f32,
) -> (Vec<usize>, Vec<f32>) {
    // Initialization
    let mut v = vec![0.0_f32; s.len()];
    for &term_state in t {
        v[term_state] = 0.0; // Set terminal state values to 0
    }
    let mut pi = vec![0; s.len()]; // Initialize policy with zeros

    // Randomly initialize the policy
    let mut rng = rand::thread_rng();
    for &state in s {
        pi[state] = rng.gen_range(0..a.len()); // Randomly select an action
    }

    loop {
        // Policy Evaluation
        v = iterative_policy_evaluation(&pi, &s, &r, &p, &t, gamma, theta);

        // Policy Improvement
        let mut stable_policy = true;

        for &state in s {
            let old_action = pi[state];
            let mut best_a = 0;
            let mut best_a_score = -9999999.99; // Initialize to negative infinity

            for &action in a {
                let mut total = 0.0;

                for &next_state in s {
                    for r_id in 0..r.len() {
                        let reward = r[r_id];
                        total += p[state][action][next_state][r_id] * (reward + gamma * v[next_state]);
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
