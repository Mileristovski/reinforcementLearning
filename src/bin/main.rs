mod cli;
mod back;
use crossterm::{
    terminal::{disable_raw_mode, enable_raw_mode},
};
use std::io::Write;
use crate::cli::elements::user_choice;
use crate::back::services::common;
use crate::back::envs;

fn main() {
    let options = vec![
        "Line World",
        "Grid World",
        "Two round Rock Paper Scissors",
        "Monty Hall \"paradox\" level 1",
        "Monty Hall \"paradox\" level 2",
        "Secret env 0",
        "Secret env 1",
        "Secret env 2",
        "Quit"
    ];

    let mut selected_index = 0;

    // Enable raw mode to capture arrow key inputs
    //enable_raw_mode().unwrap();
    //envs::grid_world_env::GridEnv::new();
    loop {
        selected_index = user_choice(options.clone());
        match selected_index {
            0 => { common::run(envs::line_world_env::LineEnv::new()); },
            1 => { common::run(envs::grid_world_env::GridEnv::new()); },
            2 => { common::run_no_dp(envs::rock_paper_scissors::RockPaperScissorsEnv::new()); }
            3 => { common::run_no_dp_dynamic(envs::monty_hall_1::MontyHallEnv::new()); }
            4 => { common::run_no_dp_dynamic(envs::monty_hall_2::MontyHallLevel2Env::new()); }
            5 => unsafe { common::run_no_dp_secret(envs::secret_env::SecretEnv::new(&format!("secret_env_{}_new", 0))); },
            6 => unsafe { common::run_no_dp_secret(envs::secret_env::SecretEnv::new(&format!("secret_env_{}_new", 1))); },
            7 => unsafe { common::run_no_dp_secret(envs::secret_env::SecretEnv::new(&format!("secret_env_{}_new", 2))); },
            8 => { break; }
            _ => {}
        }
    }
    // Disable raw mode before exiting
    disable_raw_mode().unwrap();
}
