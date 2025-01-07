use crate::back::pages::monty_hall_1::create_env;
use crate::back::rl::monte_carlo::off_policy::off_policy_mc_control_dynamic;
use crate::back::rl::monte_carlo::on_policy::on_policy_first_visit_monte_carlo_control_dynamic;
use crate::back::rl::temporal_difference_learning::q_learning::q_learning_dynamic;

pub(crate) fn testing_monte_carlo_on_policy() {
    let mut env = create_env();
    let num_episodes = 10_000;
    let epsilon = 0.1;
    let gamma = 0.99;
    let action_spaces = vec![3, 2];

    let (pi, q) = on_policy_first_visit_monte_carlo_control_dynamic(&mut env, num_episodes, epsilon, gamma, &action_spaces);

    println!("{:?}", pi);
    println!("{:?}", q);
}

pub(crate) fn testing_monte_carlo_off_policy() {
    let mut env = create_env();
    let num_episodes = 10_000;
    let gamma = 0.99;
    let action_spaces = vec![3, 2];

    let (pi, q) = off_policy_mc_control_dynamic(&mut env, num_episodes, gamma, &action_spaces);

    println!("{:?}", pi);
    println!("{:?}", q);
}
