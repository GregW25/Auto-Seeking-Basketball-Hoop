import numpy as np
from scipy.spatial.transform import Rotation as R

class FastIK:
    def __init__(self):
        # --- ROBOT SETUP (UR5e Standard DH Parameters) ---
        # If you are using a different robot, UPDATE THESE!
        # [theta, d, a, alpha]
        self.dh_params = [
            [0, 0.1625, 0, np.pi/2],       # Joint 1
            [0, 0, -0.425, 0],             # Joint 2
            [0, 0, -0.3922, 0],            # Joint 3
            [0, 0.1333, 0, np.pi/2],       # Joint 4
            [0, 0.0997, 0, -np.pi/2],      # Joint 5
            [0, 0.0996, 0, 0]              # Joint 6
        ]
        
    def get_fk(self, q):
        """ Calculates Forward Kinematics to get end-effector position """
        T = np.eye(4)
        for i, (theta_offset, d, a, alpha) in enumerate(self.dh_params):
            theta = q[i] + theta_offset
            ct, st = np.cos(theta), np.sin(theta)
            ca, sa = np.cos(alpha), np.sin(alpha)
            
            # Standard DH Transformation Matrix
            Ti = np.array([
                [ct, -st*ca,  st*sa, a*ct],
                [st,  ct*ca, -ct*sa, a*st],
                [0,   sa,     ca,    d],
                [0,   0,      0,     1]
            ])
            T = T @ Ti
        return T[:3, 3] # Return X, Y, Z

    def get_jacobian(self, q):
        """ Numerical Jacobian (Finite Difference) - Universal & Fast enough """
        delta = 1e-4
        n = len(q)
        J = np.zeros((3, n))
        
        current_pos = self.get_fk(q)
        
        for i in range(n):
            q_perturbed = np.array(q)
            q_perturbed[i] += delta
            pos_perturbed = self.get_fk(q_perturbed)
            
            # Position derivative (linear velocity)
            J[:, i] = (pos_perturbed - current_pos) / delta
            
        return J

    def solve(self, current_q, target_pos):
        """
        Newton-Raphson Iterative Solver
        Minimizes error: || FK(q) - Target ||
        """
        q = np.array(current_q)
        target = np.array(target_pos)
        
        max_iters = 20
        tolerance = 0.01 # 1cm accuracy is fine for catching
        damping = 0.05   # Damped Least Squares factor
        
        for _ in range(max_iters):
            # 1. Forward Kinematics
            current_pos = self.get_fk(q)
            error = target - current_pos
            
            # 2. Check Convergence
            if np.linalg.norm(error) < tolerance:
                return q # Success!
            
            # 3. Jacobian
            J = self.get_jacobian(q)
            
            # 4. Damped Least Squares Update: q_new = q + J_pseudo * error
            # J.T @ (J @ J.T + lambda * I)^-1 @ error
            lambda_sq = damping ** 2
            J_inv = J.T @ np.linalg.inv(J @ J.T + lambda_sq * np.eye(3))
            
            delta_q = J_inv @ error
            q += delta_q
            
        return None # Failed to converge (Target out of reach?)