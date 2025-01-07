// use crate::back::pages::monty_hall_2::create_env;
// use crate::back::rl::monte_carlo::off_policy::off_policy_mc_control_dynamic;
// use crate::back::rl::monte_carlo::on_policy::on_policy_first_visit_monte_carlo_control_dynamic;
// use crate::back::rl::temporal_difference_learning::q_learning::q_learning_dynamic;
//
// pub(crate) fn testing_q_learning() {
//     let mut env = create_env();
//     let num_episodes = 100_000;
//     let epsilon = 0.1;
//     let gamma = 0.99;
//     let action_spaces = vec![5, 4, 3, 2];
//     let alpha = 0.1;
//     let q = q_learning_dynamic(&mut env, num_episodes, alpha, epsilon, gamma, &action_spaces);
//
//     println!("\nState-Action Value Function (Q):");
//     for (state, actions) in q.iter().enumerate() {
//         let formatted_values: Vec<String> = actions
//             .iter()
//             .map(|&value| format!("{:.3}", value)) // Format values to 3 decimal places
//             .collect();
//         println!("State {}: [{}]", state, formatted_values.join(", "));
//     }
//
//     println!("-------------------------------------");
// }
//
// pub(crate) fn testing_monte_carlo_on_policy() {
//     let mut env = create_env();
//     let num_episodes = 1_000;
//     let epsilon = 0.1;
//     let gamma = 0.99;
//     let action_spaces = vec![5, 4, 3, 2];
//
//     let (pi, q) = on_policy_first_visit_monte_carlo_control_dynamic(&mut env, num_episodes, epsilon, gamma, &action_spaces);
//
//     println!("Monte Carlo On-Policy Control Results:");
//     println!("-------------------------------------");
//
//     println!("Policy (pi):");
//     for (state, actions) in pi.iter().enumerate() {
//         let formatted_actions: Vec<String> = actions
//             .iter()
//             .map(|&prob| format!("{:.3}", prob)) // Format probabilities to 3 decimal places
//             .collect();
//         println!("State {}: [{}]", state, formatted_actions.join(", "));
//     }
//
//     println!("\nState-Action Value Function (Q):");
//     for (state, actions) in q.iter().enumerate() {
//         let formatted_values: Vec<String> = actions
//             .iter()
//             .map(|&value| format!("{:.3}", value)) // Format values to 3 decimal places
//             .collect();
//         println!("State {}: [{}]", state, formatted_values.join(", "));
//     }
//
//     println!("-------------------------------------");
// }
//
// pub(crate) fn testing_monte_carlo_off_policy() {
//     let mut env = create_env();
//     let num_episodes = 1_000;
//     let gamma = 0.99;
//     let action_spaces = vec![5, 4, 3, 2];
//
//     let (pi, q) = off_policy_mc_control_dynamic(&mut env, num_episodes, gamma, &action_spaces);
//
//     println!("Monte Carlo Off-Policy Control Results:");
//     println!("-------------------------------------");
//
//     println!("Policy (pi): {:?}", &pi);
//
//     println!("\nState-Action Value Function (Q):");
//     for (state, actions) in q.iter().enumerate() {
//         let formatted_values: Vec<String> = actions
//             .iter()
//             .map(|&value| format!("{:.3}", value)) // Format values to 3 decimal places
//             .collect();
//         println!("State {}: [{}]", state, formatted_values.join(", "));
//     }
//
//     println!("-------------------------------------");
// }