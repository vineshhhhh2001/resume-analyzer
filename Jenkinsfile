pipeline {
    agent any

    environment {
        PYTHON_VERSION = '3.11'
        NODE_VERSION   = '18'
        DOCKER_IMAGE   = 'resume-analyzer'
    }

    stages {
        stage('Checkout') {
            steps {
                echo 'Checking out source code...'
                checkout scm
            }
        }

        stage('Setup Backend') {
            steps {
                echo 'Setting up Python virtual environment...'
                dir('backend') {
                    sh '''
                        python -m venv venv
                        . venv/bin/activate
                        pip install --upgrade pip
                        pip install -r requirements.txt
                        python -m spacy download en_core_web_sm
                    '''
                }
            }
        }

        stage('Backend Tests') {
            steps {
                echo 'Running backend unit tests...'
                dir('backend') {
                    sh '''
                        . venv/bin/activate
                        pytest tests/ -v --tb=short
                    '''
                }
            }
            post {
                always {
                    junit 'backend/test-results/*.xml'
                }
            }
        }

        stage('Setup Frontend') {
            steps {
                echo 'Installing frontend dependencies...'
                dir('frontend') {
                    sh 'npm ci'
                }
            }
        }

        stage('Frontend Build') {
            steps {
                echo 'Building frontend...'
                dir('frontend') {
                    sh 'npm run build'
                }
            }
        }

        stage('Docker Build') {
            steps {
                echo 'Building Docker image...'
                sh 'docker-compose build'
            }
        }

        stage('Integration Tests') {
            steps {
                echo 'Running integration tests...'
                sh '''
                    docker-compose up -d
                    sleep 10
                    curl -f http://localhost:8000/api/v1/health || exit 1
                    docker-compose down
                '''
            }
        }

        stage('Deploy') {
            when {
                branch 'main'
            }
            steps {
                echo 'Deploying to production...'
                sh 'docker-compose up -d --build'
            }
        }
    }

    post {
        always {
            echo 'Cleaning up workspace...'
            cleanWs()
        }
        success {
            echo 'Pipeline completed successfully!'
        }
        failure {
            echo 'Pipeline failed! Check logs for details.'
            mail(
                to: 'team@resume-analyzer.com',
                subject: "Jenkins Build FAILED: ${env.JOB_NAME} #${env.BUILD_NUMBER}",
                body: "Build failed. Check: ${env.BUILD_URL}"
            )
        }
    }
}
