mod rl_func;

use std::io;
use crate::back::envs::basic_env::Env;
use crate::back::envs::grid_world_env::GridEnv;
use crate::back::services::common::testing_env_manually;
use crate::cli::elements::{end_of_run, user_choice};

fn create_env() -> GridEnv {
    let rows = 3;
    let cols = 3;
    let s_vec = vec![0; rows * cols];
    let a_vec = vec![0, 1, 2, 3]; // Actions: 0=Up, 1=Down, 2=Left, 3=Right
    let r_vec = vec![0; rows * cols];
    let t_vec = vec![0, 8]; // Terminal states (e.g., top-left and bottom-right)
    GridEnv::new()
}

// fn testing_env_manually() {
//     let mut env = create_env();
//
//     while !env.is_game_over() {
//         // Display the matrix, vector, and score
//         env.display();
//         println!("Score: {}", env.score());
//
//         // Display available actions
//         let available_actions: Vec<_> = env.available_actions().iter().cloned().collect();
//         println!("Available actions: {:?}", available_actions);
//
//         // Prompt the user for input
//         println!("Enter your action (or type 'quit' to exit): ");
//
//         let mut input = String::new();
//         io::stdin()
//             .read_line(&mut input)
//             .expect("Failed to read input");
//
//         let input = input.trim();
//
//         // Check for quit command
//         if input.eq_ignore_ascii_case("quit") {
//             println!("Exiting...");
//             break;
//         }
//
//         // Parse the input as an integer
//         match input.parse::<i32>() {
//             Ok(action) => {
//                 // Perform the step
//                 if available_actions.contains(&action) {
//                     env.step(action);
//                 } else {
//                     println!("Invalid action: {}", action);
//                 }
//             }
//             Err(_) => {
//                 println!("Please enter a valid number or 'quit' to exit.");
//             }
//         }
//     }
//     println!("-------------------------------------");
//     println!("Game Over!");
//     println!("Score: {}", env.score());
//     println!("Please any key to exit...");
//     io::stdin().read_line(&mut String::new()).unwrap();
// }

pub(crate) fn run() {
    let mut selected_index = 0;
    let options = vec![
        "Manuel Test",
        "Dynamic programming : policy iteration",
        "Dynamic programming : value iteration",
        "Monte Carlo : on policy",
        "Monte Carlo : off policy",
        "Temporal difference: Q-Learning",
        "Temporal difference: SARSA",
        "Planning : Dyna-Q",
        "Back",
    ];
    loop {
        selected_index = user_choice(options.clone());
        match selected_index {
            0 => {
                let env = create_env();
                testing_env_manually(env);
            }
            1 => { println!("Not yet implemented"); }
            2 => { rl_func::testing_value_iteration() }
            3 => { rl_func::testing_monte_carlo_on_policy() }
            4 => { rl_func::testing_monte_carlo_off_policy() }
            5 => { rl_func::testing_q_learing() }
            6 => { println!("Not yet implemented"); }
            7 => { rl_func::testing_dyna_q() }
            8 => { break; }
            _ => {}
        }
    }
    // println!("Testing Policy Iteration:");
    // println!("--------------------------------------------");
    // rl_func::testing_policy_iteration();
    // println!("--------------------------------------------");
//
    // println!("Testing Value Iteration:");
    // println!("--------------------------------------------");
    // rl_func::testing_value_iteration();
    // println!("--------------------------------------------");
//
    // println!("Testing Monte Carlo on policy:");
    // println!("--------------------------------------------");
    // rl_func::testing_monte_carlo_on_policy();
    // println!("--------------------------------------------");
//
    // println!("Testing Monte Carlo off policy:");
    // println!("--------------------------------------------");
    // rl_func::testing_monte_carlo_off_policy();
    // println!("--------------------------------------------");
//
    // println!("Testing the Q Learning");
    // println!("--------------------------------------------");
    // rl_func::testing_q_learing();
    // println!("--------------------------------------------");
//
    // println!("Testing the Q Learning");
    // println!("--------------------------------------------");
    // rl_func::testing_dyna_q();
    // println!("--------------------------------------------");
}
