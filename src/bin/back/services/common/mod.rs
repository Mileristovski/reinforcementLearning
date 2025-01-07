use std::io;
use std::{thread, time::Duration};
use rand::Rng;
use crate::back::envs::basic_env::Env;
use crate::back::rl::dynamic_programming::iterative_policy_evaluation::iterative_policy_evaluation;
use crate::back::rl::dynamic_programming::policy_iteration::policy_iteration;
use crate::back::rl::dynamic_programming::value_iteration::value_iteration;
use crate::back::rl::monte_carlo::es::monte_carlo_es;
use crate::back::rl::monte_carlo::off_policy::{off_policy_mc_control_dynamic, off_policy_mc_control_secret};
use crate::back::rl::monte_carlo::on_policy::{on_policy_first_visit_monte_carlo_control, on_policy_first_visit_monte_carlo_control_dynamic, on_policy_first_visit_monte_carlo_control_secret};
use crate::back::rl::planning::dyna_q::dyna_q;
use crate::back::rl::temporal_difference_learning::q_learning::{q_learning, q_learning_dynamic};
use crate::back::rl::temporal_difference_learning::sarsa::{sarsa, sarsa_dynamic, sarsa_secret};
use crate::cli::elements::{display_pi, display_q, end_of_run, reset_screen, user_choice};

fn ask_user_for_value(prompt: &str, default: usize) -> usize {
    println!("{}", prompt);
    let mut input = String::new();
    io::stdin()
        .read_line(&mut input)
        .expect("Failed to read input");

    match input.trim().parse::<usize>() {
        Ok(value) => value,
        Err(_) => {
            println!("Invalid input. Using default value: {}", default);
            default
        }
    }
}

fn ask_user_for_float(prompt: &str, default: f64) -> f64 {
    println!("{}", prompt);
    let mut input = String::new();
    io::stdin()
        .read_line(&mut input)
        .expect("Failed to read input");

    match input.trim().parse::<f64>() {
        Ok(value) => value,
        Err(_) => {
            println!("Invalid input. Using default value: {}", default);
            default
        }
    }
}

pub fn testing_env_manually<E: Env>(env: &mut E) {
    let mut stdout = io::stdout();
    while !env.is_game_over() {
        reset_screen(&mut stdout, "");

        env.display();
        println!("Score: {}", env.score());

        let available_actions: Vec<_> = env.available_actions().iter().cloned().collect();
        println!("Available actions: {:?}", available_actions);
        println!("Enter your action (or type 'quit' to exit): ");

        let mut input = String::new();
        io::stdin()
            .read_line(&mut input)
            .expect("Failed to read input");

        let input = input.trim();
        if input.eq_ignore_ascii_case("quit") {
            println!("Exiting...");
            break;
        }

        match input.parse::<i32>() {
            Ok(action) => {
                if available_actions.contains(&action) {
                    env.step(action);
                } else {
                    println!("Invalid action: {}", action);
                }
            }
            Err(_) => {
                println!("Please enter a valid number or 'quit' to exit.");
            }
        }
    }
    reset_screen(&mut stdout, "");
    println!("-------------------------------------");
    println!("Game Over!");
    println!("Score: {}", env.score());
    env.reset();
}

pub fn test_policy<E: Env>(env: &mut E, policy: Vec<usize>) {
    let mut rng = rand::thread_rng();
    thread::sleep(Duration::from_millis(500));
    let mut stdout = io::stdout();
    println!("---------------------------------");
    println!("Testing policy on environment");
    let speed = ask_user_for_value("Enter the speed for the model in millis (default: 100): ", 100);

    env.reset();
    let mut total_score = 0.0;
    thread::sleep(Duration::from_millis(1000));

    while !env.is_game_over() {
        reset_screen(&mut stdout, "Testing environment...");

        env.display();
        println!("Score: {}", env.score());

        let available_actions: Vec<_> = env.available_actions().iter().cloned().collect();
        println!("Available actions: {:?}", available_actions);
        println!("Enter your action (or type 'quit' to exit): ");

        let state = env.state_id();
        let mut action = policy[state];
        if !available_actions.contains(&(action as i32)) {
            action = available_actions[rng.gen_range(0..available_actions.len())] as usize;
        }
        thread::sleep(Duration::from_millis(speed as u64));
        env.step(action as i32);

        total_score += env.score();
    }
    reset_screen(&mut stdout, "Testing environment...");
    println!("-------------------------------------");
    println!("Game Over! Final Score: {}", total_score);
    env.reset();
}

pub fn testing_iterative_policy_evaluation<E: Env>(env: &mut E) {
    let s = vec![0, 1, 2, 3, 4];
    let a = vec![0, 1];
    let r = vec![-1.0, 0.0, 1.0];
    let t = vec![0, 4];

    let mut pi_right = vec![vec![0.0; a.len()]; s.len()];
    // Set the second column (index 1) to 1.0
    for state in 0..s.len() {
        pi_right[state][1] = 1.0;
    }

    let mut p = vec![vec![vec![vec![0.0f32; r.len()]; s.len()]; a.len()]; s.len()];

    // Setting up the probability transition matrix p
    for &s_p in &s {
        if s_p == 0 || s_p == 4 {
            continue;
        }
        for &action in &a {
            if action == 0 && s_p > 1 {
                p[s_p][action][s_p - 1][1] = 1.0;
            }
            if action == 1 && s_p < 3 {
                p[s_p][action][s_p + 1][1] = 1.0;
            }
        }
    }

    // Set specific probabilities
    p[1][0][0][0] = 1.0;
    p[3][1][4][2] = 1.0;
    println!("{:?}", p);
    let gamma = 0.999;
    let theta = 0.0001;

    let v = iterative_policy_evaluation(&pi_right, &s, &a, &r, env, gamma, theta);
    println!("{:?}", v);
}

pub fn testing_policy_iterations<E: Env>(env: &mut E) {
    let s =  (0..env.num_states()).collect();
    let a =  (0..env.num_actions()).collect();
    let r = env.get_reward_vector();
    let t = env.get_terminal_states();
    let gamma = ask_user_for_float("Enter the gamma value (default: 0.999): ", 0.999);
    let theta = ask_user_for_float("Enter the gamma value (default: 0.0001): ", 0.0001);

    let (pi, v) = policy_iteration(&s, &a, &r, &t, env, gamma as f32, theta as f32);
    println!("Optimal Values: {:?}", v);
    println!("Optimal Policy: {:?}", pi);
    test_policy(env, pi);
}

pub fn testing_value_iteration<E: Env>(env: &mut E) {
    let s = (0..env.num_states()).collect();
    let a = (0..env.num_actions()).collect();
    let r = env.get_reward_vector();
    let gamma = ask_user_for_float("Enter the gamma value (default: 0.999): ", 0.999);
    let theta = ask_user_for_float("Enter the gamma value (default: 0.0001): ", 0.0001);

    // Call value iteration
    let (v, pi) = value_iteration(&s, &a, &r, env, gamma as f32, theta as f32);
    println!("Optimal Values: {:?}", v);
    println!("Optimal Policy: {:?}", pi);
    test_policy(env, pi);
}

pub fn testing_monte_carlo_on_policy<E: Env>(env: &mut E) {
    println!("Monte Carlo On-Policy");
    let num_episodes = ask_user_for_value("Enter the number of episodes (default: 10,000): ", 10_000);
    let epsilon = ask_user_for_float("Enter the epsilon value (default: 0.01): ", 0.01);
    let gamma = ask_user_for_float("Enter the gamma value (default: 0.999): ", 0.999);
    let pi = vec![vec![1.0 / env.num_actions() as f32; env.num_actions()]; env.num_states()];

    let (pi, q) = on_policy_first_visit_monte_carlo_control(env, num_episodes, epsilon as f32, gamma as f32, pi);

    println!("Monte Carlo On-Policy Control Results:");
    println!("-------------------------------------");

    display_pi(pi.clone());
    display_q(q);

    // Get the policy
    let pi_for_testing = pi.clone().iter()
        .map(|state_probs| {
            state_probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(index, _)| index)
                .unwrap_or(0)
        })
        .collect();

    test_policy(env, pi_for_testing);
}

pub fn testing_monte_carlo_on_policy_dynamic<E: Env>(env: &mut E) {
    println!("Monte Carlo On-Policy");
    let num_episodes = ask_user_for_value("Enter the number of episodes (default: 10,000): ", 10_000);
    let epsilon = ask_user_for_float("Enter the epsilon value (default: 0.01): ", 0.01);
    let gamma = ask_user_for_float("Enter the gamma value (default: 0.999): ", 0.999);
    let action_spaces = env.get_action_spaces();
    let (pi, q) = on_policy_first_visit_monte_carlo_control_dynamic(env, num_episodes, epsilon as f32, gamma as f32, &action_spaces);

    println!("Monte Carlo On-Policy Control Results:");
    println!("-------------------------------------");

    display_pi(pi.clone());
    display_q(q);

    // Get the policy
    let pi_for_testing = pi.clone().iter()
        .map(|state_probs| {
            state_probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(index, _)| index)
                .unwrap_or(0)
        })
        .collect();

    test_policy(env, pi_for_testing);
}

pub fn testing_monte_carlo_on_policy_secret<E: Env>(env: &mut E) {
    println!("Monte Carlo On-Policy");
    let num_episodes = ask_user_for_value("Enter the number of episodes (default: 10,000): ", 10_000);
    let epsilon = ask_user_for_float("Enter the epsilon value (default: 0.01): ", 0.01);
    let gamma = ask_user_for_float("Enter the gamma value (default: 0.999): ", 0.999);
    let (pi, q) = on_policy_first_visit_monte_carlo_control_secret(env, num_episodes, epsilon as f32, gamma as f32);

    println!("Monte Carlo On-Policy Control Results:");
    println!("-------------------------------------");

    // display_pi(pi.clone());
    // display_q(q);

    // Get the policy
    let pi_for_testing = pi.clone().iter()
        .map(|state_probs| {
            state_probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(index, _)| index)
                .unwrap_or(0)
        })
        .collect();

    test_policy(env, pi_for_testing);
}

pub fn testing_monte_carlo_off_policy<E: Env>(env: &mut E) {
    println!("Monte Carlo Off-Policy");

    let num_episodes = ask_user_for_value("Enter the number of episodes (default: 10,000): ", 10_000);
    let gamma = ask_user_for_float("Enter the gamma value (default: 0.999): ", 0.999);

    let (pi, q) = off_policy_mc_control_secret(env, num_episodes, gamma as f32);
    println!("Monte Carlo Off-Policy Control Results:");
    println!("-------------------------------------");
    println!("Policy (pi): {:?}", pi);
    // display_q(q);
    test_policy(env, pi.clone());
}

pub fn testing_monte_carlo_off_policy_dynamic<E: Env>(env: &mut E) {
    println!("Monte Carlo Off-Policy");

    let num_episodes = ask_user_for_value("Enter the number of episodes (default: 10,000): ", 10_000);
    let gamma = ask_user_for_float("Enter the gamma value (default: 0.999): ", 0.999);
    let action_spaces = env.get_action_spaces();
    let (pi, q) = off_policy_mc_control_dynamic(env, num_episodes, gamma as f32, &action_spaces);
    println!("Monte Carlo Off-Policy Control Results:");
    println!("-------------------------------------");
    println!("Policy (pi): {:?}", pi);

    display_q(q);
    test_policy(env, pi.clone());
}

pub fn testing_monte_carlo_off_policy_secret<E: Env>(env: &mut E) {
    println!("Monte Carlo Off-Policy");

    let num_episodes = ask_user_for_value("Enter the number of episodes (default: 10,000): ", 10_000);
    let gamma = ask_user_for_float("Enter the gamma value (default: 0.999): ", 0.999);
    let (pi, q) = off_policy_mc_control_secret(env, num_episodes, gamma as f32);
    println!("Monte Carlo Off-Policy Control Results:");
    println!("-------------------------------------");
    test_policy(env, pi.clone());
}

pub fn testing_monte_carlo_es<E: Env>(env: &mut E) {
    println!("Monte Carlo Off-Policy");

    let num_episodes = ask_user_for_value("Enter the number of episodes (default: 10,000): ", 10_000);
    let gamma = ask_user_for_float("Enter the gamma value (default: 0.999): ", 0.999);

    let (pi, q) = monte_carlo_es(env, num_episodes, gamma as f32);
    println!("Monte Carlo Off-Policy Control Results:");
    println!("-------------------------------------");
    println!("Policy (pi): {:?}", pi);
    display_q(q);
    test_policy(env, pi.clone());
}

pub fn testing_dyna_q<E: Env>(env: &mut E) {
    println!("Dyna-Q");
    let num_episodes = ask_user_for_value("Enter the number of episodes (default: 10,000): ", 10_000);
    let epsilon = ask_user_for_float("Enter the epsilon value (default: 0.01): ", 0.01);
    let gamma = ask_user_for_float("Enter the gamma value (default: 0.999): ", 0.999);
    let alpha = ask_user_for_float("Enter the alpha value (default: 0.01): ", 0.01);
    let planning_steps = 100;
    let (q, model) = dyna_q(env, num_episodes, alpha as f32, epsilon as f32, gamma as f32, planning_steps);

    display_q(q.clone());
    println!("Model : {:?}", model);
    let pi = q.clone().iter()
        .map(|state_probs| {
            state_probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(index, _)| index)
                .unwrap_or(0)
        })
        .collect();

    test_policy(env, pi);
}

pub fn testing_sarsa<E: Env>(env: &mut E) {
    println!("SARSA");
    let num_episodes = ask_user_for_value("Enter the number of episodes (default: 10,000): ", 10_000);
    let epsilon = ask_user_for_float("Enter the epsilon value (default: 0.01): ", 0.01);
    let gamma = ask_user_for_float("Enter the gamma value (default: 0.999): ", 0.999);
    let alpha = ask_user_for_float("Enter the alpha value (default: 0.01): ", 0.01);
    let q = sarsa(env, num_episodes, alpha as f32, epsilon as f32, gamma as f32);

    display_q(q.clone());
    let pi = q.clone().iter()
        .map(|state_probs| {
            state_probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(index, _)| index)
                .unwrap_or(0)
        })
        .collect();

    test_policy(env, pi);
}

pub fn testing_sarsa_dynamic<E: Env>(env: &mut E) {
    println!("SARSA");
    let num_episodes = ask_user_for_value("Enter the number of episodes (default: 10,000): ", 10_000);
    let epsilon = ask_user_for_float("Enter the epsilon value (default: 0.01): ", 0.01);
    let gamma = ask_user_for_float("Enter the gamma value (default: 0.999): ", 0.999);
    let alpha = ask_user_for_float("Enter the alpha value (default: 0.01): ", 0.01);
    let action_spaces = env.get_action_spaces();
    let q = sarsa_dynamic(env, num_episodes, alpha as f32, epsilon as f32, gamma as f32, &action_spaces);

    display_q(q.clone());
    let pi = q.clone().iter()
        .map(|state_probs| {
            state_probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(index, _)| index)
                .unwrap_or(0)
        })
        .collect();

    test_policy(env, pi);
}

pub fn testing_sarsa_secret<E: Env>(env: &mut E) {
    println!("SARSA");
    let num_episodes = ask_user_for_value("Enter the number of episodes (default: 10,000): ", 10_000);
    let epsilon = ask_user_for_float("Enter the epsilon value (default: 0.01): ", 0.01);
    let gamma = ask_user_for_float("Enter the gamma value (default: 0.999): ", 0.999);
    let alpha = ask_user_for_float("Enter the alpha value (default: 0.01): ", 0.01);

    let q = sarsa_secret(env, num_episodes, alpha as f32, epsilon as f32, gamma as f32);
    let pi = q.clone().iter()
        .map(|state_probs| {
            state_probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(index, _)| index)
                .unwrap_or(0)
        })
        .collect();
    // display_q(q);
    // println!("-------------------------------------");
    test_policy(env, pi)
}

pub fn testing_q_learning<E: Env>(env: &mut E) {
    println!("Q-Learning");
    let num_episodes = ask_user_for_value("Enter the number of episodes (default: 10,000): ", 10_000);
    let epsilon = ask_user_for_float("Enter the epsilon value (default: 0.01): ", 0.01);
    let gamma = ask_user_for_float("Enter the gamma value (default: 0.999): ", 0.999);
    let alpha = ask_user_for_float("Enter the alpha value (default: 0.01): ", 0.01);
    let q = q_learning(env, num_episodes, alpha as f32, epsilon as f32, gamma as f32);

    display_q(q.clone());
    let pi = q.clone().iter()
        .map(|state_probs| {
            state_probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(index, _)| index)
                .unwrap_or(0)
        })
        .collect();

    test_policy(env, pi);
}

pub fn testing_q_learning_dynamic<E: Env>(env: &mut E) {
    println!("Q-Learning");
    let num_episodes = ask_user_for_value("Enter the number of episodes (default: 10,000): ", 10_000);
    let epsilon = ask_user_for_float("Enter the epsilon value (default: 0.01): ", 0.01);
    let gamma = ask_user_for_float("Enter the gamma value (default: 0.999): ", 0.999);
    let alpha = ask_user_for_float("Enter the alpha value (default: 0.01): ", 0.01);
    let action_spaces = env.get_action_spaces();
    let q = q_learning_dynamic(env, num_episodes, alpha as f32, epsilon as f32, gamma as f32, &action_spaces);

    display_q(q.clone());
    let pi = q.clone().iter()
        .map(|state_probs| {
            state_probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(index, _)| index)
                .unwrap_or(0)
        })
        .collect();

    test_policy(env, pi);
}

pub fn testing_q_learning_secret<E: Env>(env: &mut E) {
    println!("Q-Learning");
    let num_episodes = ask_user_for_value("Enter the number of episodes (default: 10,000): ", 10_000);
    let epsilon = ask_user_for_float("Enter the epsilon value (default: 0.01): ", 0.01);
    let gamma = ask_user_for_float("Enter the gamma value (default: 0.999): ", 0.999);
    let alpha = ask_user_for_float("Enter the alpha value (default: 0.01): ", 0.01);
    let q = q_learning(env, num_episodes, alpha as f32, epsilon as f32, gamma as f32);
    let pi = q.clone().iter()
        .map(|state_probs| {
            state_probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(index, _)| index)
                .unwrap_or(0)
        })
        .collect();

    test_policy(env, pi);
}

pub fn run<E: Env>(mut env: E) {
    let mut selected_index = 0;
    let mut stdout = io::stdout();
    let options = vec![
        "Manuel Test",
        "Dynamic Programming : Policy Iteration",
        "Dynamic Programming : Value Iteration",
        "Monte Carlo : on policy",
        "Monte Carlo : off policy",
        "Temporal difference: Q-Learning",
        "Temporal difference: SARSA",
        "Planning : Dyna-Q",
        "Back",
    ];
    loop {
        selected_index = user_choice(options.clone());
        reset_screen(&mut stdout, options[selected_index]);
        match selected_index {
            0 => testing_env_manually(&mut env),
            1 => testing_policy_iterations(&mut env),
            2 => testing_value_iteration(&mut env),
            3 => testing_monte_carlo_on_policy(&mut env),
            4 => testing_monte_carlo_off_policy(&mut env),
            5 => testing_q_learning(&mut env),
            6 => testing_sarsa(&mut env),
            7 => testing_dyna_q(&mut env),
            8 => break,
            _ => {}
        }
        end_of_run();
    }
}

pub fn run_no_dp<E: Env>(mut env: E) {
    let mut selected_index = 0;
    let mut stdout = io::stdout();
    let options = vec![
        "Manuel Test",
        "Monte Carlo : on policy",
        "Monte Carlo : off policy",
        "Temporal difference: Q-Learning",
        "Temporal difference: SARSA",
        "Back"
    ];
    loop {
        selected_index = user_choice(options.clone());
        reset_screen(&mut stdout, options[selected_index]);
        match selected_index {
            0 => testing_env_manually(&mut env),
            1 => testing_monte_carlo_on_policy(&mut env),
            2 => testing_monte_carlo_off_policy(&mut env),
            3 => testing_q_learning(&mut env),
            4 => testing_sarsa(&mut env),
            5 => break,
            _ => {}
        }
        end_of_run();
    }
}

pub fn run_no_dp_dynamic<E: Env>(mut env: E) {
    let mut selected_index = 0;
    let mut stdout = io::stdout();
    let options = vec![
        "Manuel Test",
        "Monte Carlo : on policy",
        "Monte Carlo : off policy",
        "Temporal difference: Q-Learning",
        "Temporal difference: SARSA",
        "Back",
    ];
    loop {
        selected_index = user_choice(options.clone());
        reset_screen(&mut stdout, options[selected_index]);
        match selected_index {
            0 => testing_env_manually(&mut env),
            1 => testing_monte_carlo_on_policy_dynamic(&mut env),
            2 => testing_monte_carlo_off_policy_dynamic(&mut env),
            3 => testing_q_learning_dynamic(&mut env),
            4 => testing_sarsa_dynamic(&mut env),
            5 => break,
            _ => {}
        }
        end_of_run();
    }
}

pub fn run_no_dp_secret<E: Env>(mut env: E) {
    let mut selected_index = 0;
    let mut stdout = io::stdout();
    let options = vec![
        "Manuel Test",
        "Monte Carlo : on policy",
        "Monte Carlo : off policy",
        "Temporal difference: Q-Learning",
        "Temporal difference: SARSA",
        "Back",
    ];
    loop {
        selected_index = user_choice(options.clone());
        reset_screen(&mut stdout, options[selected_index]);
        match selected_index {
            0 => testing_env_manually(&mut env),
            1 => testing_monte_carlo_on_policy_secret(&mut env),
            2 => testing_monte_carlo_off_policy_secret(&mut env),
            3 => testing_q_learning_secret(&mut env),
            4 => testing_sarsa_secret(&mut env),
            5 => break,
            _ => {}
        }
        end_of_run();
    }
}