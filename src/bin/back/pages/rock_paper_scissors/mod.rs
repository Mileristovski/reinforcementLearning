mod rl_func;

use crate::back::envs::rock_paper_scissors::RockPaperScissorsEnv;
use crate::back::envs::basic_env::Env;
use crate::back::services::common;
use crate::cli::elements::{end_of_run, reset_screen, user_choice};

fn create_env() -> RockPaperScissorsEnv {
    RockPaperScissorsEnv::new()
}

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
}
