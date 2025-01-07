use nalgebra::{DVector};
#[allow(dead_code)]
pub trait Env {
    fn num_states(&self) -> usize;
    fn num_actions(&self) -> usize;
    fn num_rewards(&self) -> usize;
    fn reward(&self, i: usize) -> f32;
    fn p(&self, s: i32, a: i32, s_p: i32, r_index: i32) -> f32;
    fn state_id(&self) -> usize;
    fn reset(&mut self);
    fn display(&self);
    fn is_forbidden(&self, action: usize) -> bool;
    fn is_game_over(&self) -> bool;
    fn available_actions(&self) -> DVector<i32>;
    fn step(&mut self, action: i32);
    fn score(&self) -> f32;
    fn from_random_state() -> Self where Self: Sized;
}