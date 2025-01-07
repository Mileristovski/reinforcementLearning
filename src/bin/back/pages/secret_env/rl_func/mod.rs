use std::fs;
use crate::back::envs::secret_env::SecretEnv;
use crate::back::envs::basic_env::Env;
use crate::back::rl::dynamic_programming::policy_iteration::policy_iteration_secret;
use crate::back::rl::monte_carlo::on_policy::on_policy_first_visit_monte_carlo_control_secret;

pub(crate) unsafe fn testing_policy_iterations(num: String) {
    let mut env = SecretEnv::new(&format!("secret_env_{}_new", num));
    let s: Vec<_> = (0..env.num_states()).collect();
    let a: Vec<_> = (0..env.num_actions()).collect();
    let r: Vec<_> = (0..env.num_rewards()).map(|i| env.reward(i)).collect();
    let gamma = 0.9;
    let theta = 0.1;


    // Call policy iteration
    let v = policy_iteration_secret(&s, &a, &r, &mut env, gamma, theta);
    let serialized = serde_json::to_string(&v).expect("Serialization failed");
    fs::write("./tmp/output", serialized).expect("Unable to write file");
    println!("{:?}", v);

    env.delete();
}

pub(crate) unsafe fn testing_monte_carlo_on_policy(num: String) {
    let mut env = SecretEnv::new(&format!("secret_env_{}_new", num));
    let num_episodes = 10_000;
    let epsilon = 0.1;
    let gamma = 0.99;

    let (pi, q) = on_policy_first_visit_monte_carlo_control_secret(&mut env, num_episodes, epsilon, gamma);

    println!("Monte Carlo On-Policy Control Results:");
    println!("-------------------------------------");

    println!("Policy (pi):");
    for (state, actions) in pi.iter().enumerate() {
        let formatted_actions: Vec<String> = actions
            .iter()
            .map(|&prob| format!("{:.3}", prob)) // Format probabilities to 3 decimal places
            .collect();
        println!("State {}: [{}]", state, formatted_actions.join(", "));
    }

    println!("\nState-Action Value Function (Q):");
    for (state, actions) in q.iter().enumerate() {
        let formatted_values: Vec<String> = actions
            .iter()
            .map(|&value| format!("{:.3}", value)) // Format values to 3 decimal places
            .collect();
        println!("State {}: [{}]", state, formatted_values.join(", "));
    }

    println!("-------------------------------------");

    env.delete();
}
