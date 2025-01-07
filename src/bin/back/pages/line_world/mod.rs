mod rl_func;

use std::io::Write;
use crate::back::envs::line_world_env::LineEnv;
use crate::back::envs::basic_env::Env;
use crate::back::services::common;
use crate::cli::elements::{user_choice, end_of_run, reset_screen};

fn create_env() -> LineEnv {
    // Example usage
    let s = vec![0, 1, 2, 3, 4]; // States (as indices)
    let a = vec![0, 1]; // Actions
    let r = vec![-1, 0, 1]; // Rewards
    let t = vec![0, 4]; // Terminal states

    // Create an instance of LineEnv
    let env = LineEnv::new(s.clone(), a.clone(), r.clone(), t.clone());
    env
}
pub fn run() {
    let mut selected_index = 0;
    let mut stdout = std::io::stdout();
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
        reset_screen(&mut stdout, "");
        let env = create_env();
        match selected_index {
            0 => common::testing_env_manually(env),
            1 => rl_func::testing_policy_iterations(),
            2 => rl_func::testing_value_iteration(),
            3 => common::testing_monte_carlo_on_policy(env),
            4 => common::testing_monte_carlo_off_policy(env),
            5 => common::testing_q_learning(env),
            6 => common::testing_sarsa(env),
            7 => common::testing_dyna_q(env),
            8 => break,
            _ => {}
        }
        end_of_run();
    }
}
