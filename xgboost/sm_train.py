import argparse
from sagemaker.estimator import Estimator


# You can set a role here if you want but it's not really necessary if you're running locally.
def get_args():
    default_role = 'arn:aws:iam::111122223333:role/role-name'   # Dummy default role that works.
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--role', help='Add role', default=default_role)
    parser.add_argument('-d', '--docker', help='Add your docker image name here you built locally')
    args = parser.parse_args()
    return args


def train_sm():

    args = get_args()
    
    role = args.role
    image_uri = args.docker

    output_path = 'file:///tmp/model/'

    estimator = Estimator(image_uri=image_uri,
                        role=role,
                        train_instance_count=1,
                        train_instance_type='local', 
                        output_path=output_path)


    input_data = {
        'training' : 'file://dataset/train/',
        'testing' : 'file://dataset/testing/',
    }

    estimator.fit(inputs=input_data)


if __name__ == "__main__":
    train_sm()