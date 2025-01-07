use crate::back::pages::rock_paper_scissors::create_env;
use crate::back::envs::basic_env::Env;
use crate::back::rl::monte_carlo::off_policy::off_policy_mc_control;
use crate::back::rl::monte_carlo::on_policy::on_policy_first_visit_monte_carlo_control;
use crate::back::rl::temporal_difference_learning::q_learning::q_learning;

pub(crate) fn testing_monte_carlo_on_policy() {
    let mut env = create_env();
    let num_episodes = 10_000;
    let epsilon = 0.1;
    let gamma = 0.99;
    let pi = vec![vec![1.0 / env.num_actions() as f32; env.num_actions()]; env.num_states()];

    let (pi, q) = on_policy_first_visit_monte_carlo_control(&mut env, num_episodes, epsilon, gamma, pi);

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
}

pub(crate) fn testing_monte_carlo_off_policy() {
    let mut env = create_env();
    let num_episodes = 10_000;
    let gamma = 0.99;

    let (pi, q) = off_policy_mc_control(&mut env, num_episodes, gamma);

    println!("Monte Carlo Off-Policy Control Results:");
    println!("-------------------------------------");

    println!("Policy (pi): {:?}", &pi);

    println!("\nState-Action Value Function (Q):");
    for (state, actions) in q.iter().enumerate() {
        let formatted_values: Vec<String> = actions
            .iter()
            .map(|&value| format!("{:.3}", value)) // Format values to 3 decimal places
            .collect();
        println!("State {}: [{}]", state, formatted_values.join(", "));
    }

    println!("-------------------------------------");
}

pub(crate) fn testing_q_learning() {
    let mut env = create_env();
    let num_episodes = 10_000;
    let epsilon = 0.1;
    let gamma = 0.99;
    let alpha = 0.1;

    let q = q_learning(&mut env, num_episodes, alpha, epsilon, gamma);

    println!("Monte Carlo On-Policy Control Results:");
    println!("-------------------------------------");

    println!("\nState-Action Value Function (Q):");
    for (state, actions) in q.iter().enumerate() {
        let formatted_values: Vec<String> = actions
            .iter()
            .map(|&value| format!("{:.3}", value)) // Format values to 3 decimal places
            .collect();
        println!("State {}: [{}]", state, formatted_values.join(", "));
    }

    println!("-------------------------------------");
}