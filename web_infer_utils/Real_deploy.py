#!/usr/bin/env python3
"""
Real Deploy Server 
"""
from flask import Flask, request, Response, jsonify
import argparse
import os
import json
import sys
import logging
import cv2
import time
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
root_dir = os.path.dirname(current_dir)
sys.path.insert(0, root_dir)
import torch
import numpy as np
from web_infer_utils.MVActor import MVActor


def decode_uint8_image(raw_bytes, shape, field_name):
    """Decode raw image bytes and tolerate a single missing byte."""
    expected_size = int(np.prod(shape))
    actual_size = len(raw_bytes)

    if actual_size == expected_size - 1 and actual_size > 0:
        logger.warning(
            "%s bytes truncated by 1 byte; padding with the previous byte for robustness",
            field_name,
        )
        raw_bytes += raw_bytes[-1:]
    elif actual_size != expected_size:
        raise ValueError(
            f"{field_name} byte size mismatch: expected {expected_size}, got {actual_size}"
        )

    return np.frombuffer(raw_bytes, dtype=np.uint8).reshape(shape)


class RobotServer:

    def __init__(
        self,
        config_file,
        transformer_file,
        device=0,
        denoise_step=10,
        threshold=1,
        action_dim=14,
        gripper_dim=1,
        domain_name="robotwin",
        norm_type="meanstd",
        n_prev=4,
    ):
        """
        Args:
            config_file: Path to the YAML configuration file
            transformer_file: Model weight paths
            device: GPU ID
            denoise_step: Number of denoise steps
            threshold: Number of steps to update historical frames
            action_dim: Action dimensions(include grippers)
            gripper_dim: Gripper dimensions
            domain_name: Dataset Field Name
            n_prev: Historical Frames
        """
        logger.info(f"Initializing RobotServer with config: {config_file}")
        logger.info(f"Transformer file: {transformer_file}")
        logger.info(f"Device: cuda:{device}")

        
        device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")

        self.actor = MVActor(
            config_file=config_file,
            transformer_file=transformer_file,
            threshold=threshold,
            n_prev=n_prev,
            action_dim=action_dim,
            gripper_dim=gripper_dim,
            domain_name=domain_name,
            load_weights=True,
            num_inference_steps=denoise_step,
            device=device,
            dtype=torch.bfloat16,
            norm_type=norm_type,
        )
        self.action_dim = action_dim
        self.gripper_dim = gripper_dim
        self.arm_dim = (action_dim - 2 * gripper_dim) // 2

        logger.info("RobotServer initialized successfully")

    def predict(self, obs, prompt, state=None, execution_step=1):
        """

        Args:
            obs: shape [v, c, h, w] or [v, h, w, c]
            prompt
            state
            execution_step: Number of execution steps

        Returns:
            action
        """
        

        action = self.actor.play(
            obs=obs,
            prompt=prompt,
            state=state,
            execution_step=execution_step,
            explore_config = {
                "explore_steps": 1,
                "dynamic_groups": 30,
                "value_groups": 5,
                "sigma_decay": 0.5,
                "alpha_smooth": 0.9,
                "value_elites": 0.1,
                "dynamic_elites": 0.9,
            },
            state_zeropadding=[14,0]
        )
        print(f"Action {action}")
        return action


def create_app(server: RobotServer):
    app = Flask(__name__)

    @app.route("/healthz", methods=["GET"])
    def healthz():
        return Response("OK\n", content_type='text/plain', status=200)

    @app.route("/predict", methods=["POST"])
    def predict():

        start_time = __import__('time').perf_counter_ns()

        try:
            front_head = decode_uint8_image(
                request.files["front_head"].read(), (240, 424, 3), "front_head"
            ) #480,424

            left_hand = decode_uint8_image(
                request.files["left_hand"].read(), (240, 320, 3), "left_hand"
            ) #240, 320

            right_hand = decode_uint8_image(
                request.files["right_hand"].read(), (240, 320, 3), "right_hand"
            ) #240, 320

            save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_images_real")
            os.makedirs(save_dir, exist_ok=True)

            timestamp = int(time.time() * 1000)

            cv2.imwrite(os.path.join(save_dir, f"{timestamp}_front_head.png"), front_head)
            cv2.imwrite(os.path.join(save_dir, f"{timestamp}_left_hand.png"), left_hand)
            cv2.imwrite(os.path.join(save_dir, f"{timestamp}_right_hand.png"), right_hand)

            target_size=(256,256)

            front_head_image_r = cv2.resize(front_head,target_size,interpolation=cv2.INTER_LINEAR)

            left_hand_image_r = cv2.resize(left_hand,target_size,interpolation=cv2.INTER_LINEAR)


            right_hand_image_r = cv2.resize(right_hand,target_size,interpolation=cv2.INTER_LINEAR)



            cv2.imwrite(os.path.join(save_dir, f"{timestamp}_front_head_r.png"), front_head_image_r)
            cv2.imwrite(os.path.join(save_dir, f"{timestamp}_left_hand_r.png"), left_hand_image_r)
            cv2.imwrite(os.path.join(save_dir, f"{timestamp}_right_hand_r.png"), right_hand_image_r)



            obs_list = [front_head_image_r, left_hand_image_r, right_hand_image_r]
            obs = np.stack([img for img in obs_list], axis=0)  # [v, H, W,3]

            state = np.frombuffer(
                request.files["state"].read(), dtype=np.float32
            ).reshape((14))


            content = json.loads(request.files["json"].read())
            instruction = content["instruction"]


            action = server.predict(
                obs=obs,
                prompt=instruction,
                state=state,
                execution_step=54,
            )

            action_bytes = action.tobytes()

            end_time = __import__('time').perf_counter_ns()
            logger.info(
                f"total inference time: {(end_time - start_time) / 1e6} ms"
            )

            return Response(action_bytes, content_type='application/octet-stream')

        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500

    @app.route("/reset", methods=["POST"])
    def reset():
        """Reset server status"""
        server.actor.reset()
        return Response("Reset OK\n", content_type='text/plain', status=200)

    return app


def main():
    parser = argparse.ArgumentParser(description="Real Deploy Server")

    parser.add_argument(
        '-c', '--config',
        type=str,
        default='configs/ltx_model/robotwin/policy_model_rel_joint_robotwin.yaml',
        help='Path to the YAML model config'
    )

    parser.add_argument(
        '-w', '--weight',
        type=str,
        default='/defaultShare/Genie-Envisioner/lrz_training_v2/robotwin4/Phase3_PolicyModel_WOV_rel_joint_54/2026_02_22_19_12_11/step_20000/diffusion_pytorch_model.safetensors',
        help='Path to the model weight'
    )

    # Server Configuration
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Server host')
    parser.add_argument('-p', '--port', type=int, default=8001, help='Server port')

    # Model parameters
    parser.add_argument('--domain_name', type=str, default='robotwin', help='Domain name')
    parser.add_argument('--threshold', type=float, default=1, help='Memory update threshold')
    parser.add_argument('--denoise_step', type=int, default=10, help='Denoising steps')
    parser.add_argument('--action_dim', type=int, default=14, help='Action dimension')
    parser.add_argument('--norm_type', type=str, default="meanstd")
    parser.add_argument('--device', type=int, default=0, help='GPU device ID')

    args = parser.parse_args()

    logger.info("=" * 50)
    logger.info("Starting Real Deploy Server")
    logger.info(f"Config: {args.config}")
    logger.info(f"Weight: {args.weight}")
    logger.info(f"Port: {args.port}")
    logger.info("=" * 50)

    server = RobotServer(
        config_file=args.config,
        transformer_file=args.weight,
        device=args.device,
        denoise_step=args.denoise_step,
        threshold=args.threshold,
        action_dim=args.action_dim,
        domain_name=args.domain_name,
        norm_type=args.norm_type
    )

    app = create_app(server)

    logger.info(f"Server starting on {args.host}:{args.port}")
    logger.info(f"Health check: http://{args.host}:{args.port}/healthz")
    logger.info(f"Prediction endpoint: http://{args.host}:{args.port}/predict")

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
