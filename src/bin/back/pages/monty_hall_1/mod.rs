mod rl_func;

use crate::back::envs::monty_hall_1::MontyHallEnv;
use crate::back::envs::basic_env::Env;
use crate::back::services::common;
use crate::cli::elements::{end_of_run, reset_screen, user_choice};

fn create_env() -> MontyHallEnv {
    MontyHallEnv::new()
}

// fn testing_env_manually() {
//     let mut env = create_env();
//
//     println!("Testing Monty Hall Environment");
//     println!("---------------------------------------------");
//
//     env.display();
//     println!("Score: {}", env.score());
//
//     // Step 1: Choose a door (e.g., Door 1)
//     println!("Step 1: Agent chooses Door 1 (action = 1)");
//     println!("Available actions: {}", env.available_actions());
//     env.step(2); // Choose Door 1
//     env.display();
//
//     // Step 2: Decide to switch (action = 1) or keep (action = 0)
//     println!("Step 2: Agent decides to switch doors (action = 1)");
//     println!("Available actions: {}", env.available_actions()[1]);
//     env.step(1); // Switch
//     env.display();
//
//     println!("Game Over? {}", env.is_game_over());
//     println!("Final Score: {}", env.score());
// }

pub fn run() {
    let mut selected_index = 0;
    let mut stdout = std::io::stdout();
    let options = vec![
        "Manuel Test",
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
            1 => common::testing_monte_carlo_on_policy(env),
            2 => common::testing_monte_carlo_off_policy(env),
            3 => common::testing_q_learning(env),
            4 => common::testing_sarsa(env),
            5 => common::testing_dyna_q(env),
            6 => break,
            _ => {}
        }
        end_of_run();
    }
    // let env = create_env();
    // println!("Testing the Monty Hall environment manually");
    // println!("-------------------------------------------------");
    // testing_env_manually(env);
    // println!("-------------------------------------------------");
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
    // rl_func::testing_q_learning();
    // println!("--------------------------------------------");
}
