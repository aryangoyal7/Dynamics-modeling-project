T4 pick-and-place (control task): grasp the blue cube and hold it at the goal point.
Quadrants (row-major):
  0: nominal physics
  1: light + slippery (overshoots)
  2: heavy + rough (stops short)
  3: off-centre weight (veers/rotates)
Controller: trained privileged PPO teacher. Once grasped, hidden physics stop mattering - that is the point of this task.
