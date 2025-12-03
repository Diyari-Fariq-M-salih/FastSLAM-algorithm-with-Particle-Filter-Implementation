from robot_simulator import RobotSim
sim = RobotSim()
z, gt = sim.commandAndGetData(3, 6)
print(z.shape)