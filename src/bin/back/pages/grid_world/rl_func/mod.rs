use crate::back::envs::grid_world_env::GridEnv;
use crate::back::envs::basic_env::Env;
use crate::back::pages::grid_world::create_env;
use crate::back::rl::dynamic_programming::policy_iteration::policy_iteration;
use crate::back::rl::dynamic_programming::value_iteration::value_iteration;
use crate::back::rl::monte_carlo::off_policy::off_policy_mc_control;
use crate::back::rl::monte_carlo::on_policy::on_policy_first_visit_monte_carlo_control;
use crate::back::rl::planning::dyna_q::dyna_q;
use crate::back::rl::temporal_difference_learning::q_learning::q_learning;

fn create_transition_matrix(env: &GridEnv) -> Vec<Vec<Vec<Vec<f32>>>> {
    let num_states = env.num_states();
    let num_actions = env.num_actions();
    let mut p = vec![vec![vec![vec![0.0; env.num_rewards()]; num_states]; num_actions]; num_states];

    for s in 0..num_states {
        if env.t.contains(&s) {
            continue; // Skip terminal states
        }

        let (row, col) = env.index_to_rc(s);

        for a in 0..num_actions {
            let (new_row, new_col) = match a {
                0 if row > 0 => (row - 1, col), // Up
                1 if row < env.rows - 1 => (row + 1, col), // Down
                2 if col > 0 => (row, col - 1), // Left
                3 if col < env.cols - 1 => (row, col + 1), // Right
                _ => (row, col), // No movement
            };

            let new_state = env.rc_to_index(new_row, new_col);

            // Set the probability for transitioning to the new state
            p[s][a][new_state][0] = 1.0; // Assume deterministic transitions
        }
    }
    p
}

fn create_reward_vector(env: &GridEnv) -> Vec<f32> {
    let num_states = env.num_states();
    let mut rewards = vec![0.0; num_states];

    for s in 0..num_states {
        if s == 0 {
            rewards[s] = -1.0; // Losing state
        } else if s == env.num_states() - 1 {
            rewards[s] = 10.0; // Winning state
        } else {
            rewards[s] = -0.1;
        }
    }
    rewards
}

pub(crate) fn testing_policy_iteration() {
    let env = create_env();
    dbg!(&env.num_states());
    let s: Vec<usize> = (0..env.num_states()).collect();
    let a: Vec<usize> = (0..env.num_actions()).collect();
    let t: Vec<usize> = env.t.clone();
    let r = create_reward_vector(&env);
    let p = create_transition_matrix(&env);

    let gamma = 0.99;
    let theta = 0.01;

    let (pi, v) = policy_iteration(&s, &a, &r, &p, &t, gamma, theta);
    println!("Policy Iteration Results:");
    println!("Policy: {:?}", pi);
    println!("Value Function: {:?}", v);
}

pub(crate) fn testing_value_iteration() {
    let env = create_env();
    let s: Vec<usize> = (0..env.num_states()).collect();
    let a: Vec<usize> = (0..env.num_actions()).collect();
    let t: Vec<usize> = env.t.clone();
    let r = create_reward_vector(&env);
    let p = create_transition_matrix(&env);

    let gamma = 0.999;
    let theta = 0.0001;

    let (v, pi) = value_iteration(&s, &a, &r, &p, &t, gamma, theta);
    println!("Value Iteration Results:");
    println!("Value Function: {:?}", v);
    println!("Policy: {:?}", pi);
}

pub(crate) fn testing_monte_carlo_on_policy() {
    let mut env = create_env();
    let num_episodes = 10_000;
    let epsilon = 0.001;
    let gamma = 0.999;
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
    let num_episodes = 1_000_000;
    let gamma = 0.9999;

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

pub(crate) fn testing_dyna_q() {
    let mut env = create_env();
    let num_episodes = 100_000;
    let epsilon = 0.01;
    let gamma = 0.999;
    let alpha = 0.001;
    let planning_steps = 100;
    let (q, model) = dyna_q(&mut env, num_episodes, alpha, epsilon, gamma, planning_steps);

    println!("\nState-Action Value Function (Q):");
    for (state, actions) in q.iter().enumerate() {
        let formatted_values: Vec<String> = actions
            .iter()
            .map(|&value| format!("{:.3}", value)) // Format values to 3 decimal places
            .collect();
        println!("State {}: [{}]", state, formatted_values.join(", "));
    }

    println!("-------------------------------------");
    println!("Model : {:?}", &model);
}

pub(crate) fn testing_q_learing() {
    let mut env = create_env();
    let num_episodes = 100_000;
    let epsilon = 0.1;
    let gamma = 0.99;
    let alpha = 0.1;
    let q = q_learning(&mut env, num_episodes, alpha, epsilon, gamma);

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