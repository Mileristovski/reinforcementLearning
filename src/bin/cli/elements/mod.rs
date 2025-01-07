use std::io;
use std::io::Stdout;
use std::process::Command;
use crossterm::event::{read, Event, KeyCode};
use crossterm::execute;
use crossterm::terminal::{Clear, ClearType};

pub fn esgi_logo() {
    let logo = r#"
    * __________________________*
    *   _____ ____   ____ ___   *
    *  | ____/ ___| / ___|_ _|  *
    *  |  _| \___ \| |  _ | |   *
    *  | |___ ___) | |_| || |   *
    *  |_____|____/ \____|___|  *
    * __________________________*
     "#;

    println!("{}", logo);
}

pub fn reset_screen(stdout: &mut Stdout, message: &str) {
    clear_screen();
    // Clear screen and draw the logo and menu
    execute!(stdout, Clear(ClearType::All)).unwrap();
    esgi_logo();
    println!("----------------------------------------------------------------------");
    println!("{}", message);
    println!("----------------------------------------------------------------------");
}

pub fn end_of_run() {
    println!("\nPlease any key to exit...");
    io::stdin().read_line(&mut String::new()).unwrap();
}

fn clear_screen() {
    if cfg!(target_os = "windows") {
        Command::new("cmd")
            .args(&["/C", "cls"])
            .status()
            .unwrap();
    } else {
        Command::new("clear")
            .status()
            .unwrap();
    }
}

pub fn display_pi(pi: Vec<Vec<f32>>) {
    println!("Policy (pi): {:?}", pi);
    for (state, actions) in pi.iter().enumerate() {
        let formatted_actions: Vec<String> = actions
            .iter()
            .map(|&prob| format!("{:.3}", prob)) // Format probabilities to 3 decimal places
            .collect();
        println!("State {}: [{}]", state, formatted_actions.join(", "));
    }
}

pub fn display_q(q: Vec<Vec<f32>>) {
    println!("\nState-Action Value Function (Q):");
    for (state, actions) in q.iter().enumerate() {
        let formatted_values: Vec<String> = actions
            .iter()
            .map(|&value| format!("{:.3}", value)) // Format values to 3 decimal places
            .collect();
        println!("State {}: [{}]", state, formatted_values.join(", "));
    }
}

pub fn user_choice(options: Vec<&str>) -> usize {
    let mut stdout = io::stdout();
    let mut selected_index = 0;
    loop {
        reset_screen(&mut stdout, "Use Arrow Keys to Navigate, Enter to Select: \n");
        // Display menu options
        for (i, option) in options.iter().enumerate() {
            if i == selected_index {
                println!("> {}", option);
            } else {
                println!("  {}", option);
            }
        }

        let event = read().unwrap();
        if event == Event::Key(KeyCode::Up.into()) {
            if selected_index > 0 {
                selected_index -= 1;
            } else if selected_index == 0 {
                selected_index = options.len() - 1
            }

        } else if event == Event::Key(KeyCode::Down.into()) {
            if selected_index < options.len() - 1 {
                selected_index += 1;
            } else if selected_index == options.len() - 1 {
                selected_index = 0;
            }

        } else if event == Event::Key(KeyCode::Enter.into()) {
            return selected_index;
        } else if event == Event::Key(KeyCode::Esc.into()) {
            return selected_index;
        }
    }
}