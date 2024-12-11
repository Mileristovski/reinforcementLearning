mod envs;
mod rl;

use envs::line_world_env::LineEnv;
use envs::line_world_env::Env;
use rl::dynamic_programming::policy_iteration::policy_iteration;
use rl::dynamic_programming::iterative_policy_eval::iterative_policy_evaluation;
use rl::monte_carlo::on_policy_first_visit_monte_carlo_control;

fn create_env() -> LineEnv {
    // Example usage
    let s = vec![0, 1, 2, 3, 4]; // States (as indices)
    let a = vec![0, 1]; // Actions
    let r = vec![-1, 0, 1]; // Rewards
    let t = vec![0, 4]; // Terminal states

    // Create an instance of LineEnv
    LineEnv::new(s.clone(), a.clone(), r.clone(), t.clone())
}

fn testing_env_manually() {
    let mut env = create_env();

    // Display matrix and vector
    env.display();
    println!("Score : {}", env.score());

    env.step(0);
    env.display();
    println!("Score : {}", env.score());

    env.step(1);
    env.display();
    println!("Score : {}", env.score());

    env.step(1);
    env.display();
    println!("Score : {}", env.score());

    env.step(1);
    env.display();
    println!("Score : {}", env.score());
}

fn testing_iterative_policy_evaluation() {
    let s = vec![0, 1, 2, 3, 4]; // States (as indices)
    let a = vec![0, 1]; // Actions
    let r = vec![-1.0, 0.0, 1.0]; // Rewards
    let t = vec![0, 4]; // Terminal states

    let mut pi_right = vec![vec![0.0; a.len()]; s.len()];
    // Set the second column (index 1) to 1.0
    for state in 0..s.len() {
        pi_right[state][1] = 1.0;
    }

    let mut p = vec![vec![vec![vec![0.0f32; r.len()]; s.len()]; a.len()]; s.len()]; // Transition probabilities

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

    let gamma = 0.999;
    let theta = 0.0001;

    let v = iterative_policy_evaluation(&pi_right, &s, &a, &r, &p, &t, gamma, theta);
    println!("{:?}", v);
}

fn testing_policy_iterations() {
    let s = vec![0usize, 1, 2, 3, 4];
    let a = vec![0usize, 1];
    let r = vec![-1.0f32, 0.0, 1.0];
    let t = vec![0usize, 4];

    let mut p = vec![vec![vec![vec![0.0f32; r.len()]; s.len()]; a.len()]; s.len()]; // Transition probabilities

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

    let gamma = 0.999;
    let theta = 0.0001;

    // Call policy iteration
    let v = policy_iteration(&s, &a, &r, &p, &t, gamma, theta);
    println!("{:?}", v);
}

fn testing_monte_carlo_on_policy() {
    let mut env = create_env();

    let res = on_policy_first_visit_monte_carlo_control(&mut env, 1_000, 0.1, 0.999);
    println!("{:?}", &res);
}

fn main() {
    println!("Testing the Line World environment manually");
    println!("--------------------------------------------");
    testing_env_manually();
    println!("--------------------------------------------");

    println!("Testing the policy iterations evaluation");
    println!("--------------------------------------------");
    testing_iterative_policy_evaluation();
    println!("--------------------------------------------");

    println!("Testing the policy iterations");
    println!("--------------------------------------------");
    testing_policy_iterations();
    println!("--------------------------------------------");

    println!("Testing the MonteCarlo method");
    println!("--------------------------------------------");
    testing_monte_carlo_on_policy();
    println!("--------------------------------------------");
}
