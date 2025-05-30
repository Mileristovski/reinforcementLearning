use crate::back::envs::basic_env::Env;

pub fn iterative_policy_evaluation(
    pi: &Vec<Vec<f32>>,
    s: &Vec<usize>,
    a: &Vec<usize>,
    r: &Vec<f32>,
    env: &mut dyn Env,
    gamma: f32,
    theta: f32
) -> Vec<f32> {
    let mut v = vec![0.0_f32; s.len()];

    loop {
        let mut delta: f32 = 0.0;

        for &state in s {
            let v_old = v[state];
            let mut total = 0.0;
            for &action in a {
                let mut action_value = 0.0;
                for &next_state in s {
                    for r_id in 0..r.len() {
                        action_value += env.transition_probability(state, action, next_state, r_id) * (r[r_id] + gamma * v[next_state]);
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