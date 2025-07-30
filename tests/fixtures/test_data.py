"""
Test fixtures and sample data for comprehensive testing

Provides realistic test data for emails, social media posts, files, and other data types.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any
import random
import uuid


class TestDataGenerator:
    """Generate realistic test data for testing scenarios"""
    
    @staticmethod
    def generate_email_messages(count: int = 10) -> List[Dict[str, Any]]:
        """Generate realistic email messages for testing"""
        subjects = [
            "Weekly Team Update",
            "Project Alpha - Next Steps",
            "Meeting Follow-up: Q3 Review",
            "Urgent: Server Maintenance Tonight",
            "Welcome to the Team!",
            "Re: Budget Approval Request",
            "Invoice #12345 - Payment Due",
            "Conference Registration Confirmation",
            "Happy Birthday!",
            "System Alert: Backup Completed"
        ]
        
        senders = [
            ("John Smith", "john.smith@company.com"),
            ("Sarah Johnson", "sarah.j@example.org"),
            ("Mike Davis", "mike.davis@tech.com"),
            ("Lisa Chen", "l.chen@startup.io"),
            ("Robert Wilson", "rwilson@corp.net"),
            ("Emma Thompson", "emma@freelance.com"),
            ("David Brown", "d.brown@university.edu"),
            ("Anna Garcia", "a.garcia@nonprofit.org"),
            ("James Lee", "james.lee@consulting.biz"),
            ("Maria Rodriguez", "maria.r@agency.co")
        ]
        
        body_templates = [
            "Hi {recipient},\n\nI hope this email finds you well. {main_content}\n\nPlease let me know if you have any questions.\n\nBest regards,\n{sender_name}",
            "Hello {recipient},\n\n{main_content}\n\nThanks for your time.\n\n{sender_name}",
            "Dear {recipient},\n\nI wanted to follow up on {main_content}\n\nLooking forward to your response.\n\nSincerely,\n{sender_name}",
            "{recipient},\n\n{main_content}\n\nLet me know your thoughts.\n\n- {sender_name}"
        ]
        
        main_contents = [
            "I wanted to update you on the progress of our current project. We've completed the initial phase and are moving into testing.",
            "Could we schedule a meeting next week to discuss the budget allocation for Q4?",
            "The server maintenance has been completed successfully. All systems are back online.",
            "I've attached the quarterly report for your review. Please provide feedback by Friday.",
            "We need to discuss the timeline for the upcoming product launch.",
            "The team meeting has been rescheduled to Thursday at 2 PM in Conference Room A.",
            "I'm excited to announce that we've reached our sales target for this quarter!",
            "Please find the contract attached. Let me know if you need any modifications.",
            "The training session went very well. Here are the key takeaways for the team.",
            "I wanted to share some insights from the industry conference I attended last week."
        ]
        
        emails = []
        for i in range(count):
            sender_name, sender_email = random.choice(senders)
            subject = random.choice(subjects)
            main_content = random.choice(main_contents)
            body_template = random.choice(body_templates)
            
            recipient_email = "user@example.com"
            direction = random.choice(["sent", "received"])
            
            if direction == "sent":
                from_email = recipient_email
                from_name = "Test User"
                to_emails = [sender_email]
            else:
                from_email = sender_email
                from_name = sender_name
                to_emails = [recipient_email]
            
            body = body_template.format(
                recipient="there" if direction == "sent" else from_name.split()[0],
                main_content=main_content,
                sender_name=from_name if direction == "sent" else sender_name
            )
            
            email = {
                'id': f'email_{i+1}_{uuid.uuid4().hex[:8]}',
                'subject': subject,
                'body': body,
                'snippet': body[:150] + "..." if len(body) > 150 else body,
                'from': from_email,
                'from_name': from_name,
                'to': to_emails,
                'cc': [],
                'bcc': [],
                'date': datetime.now() - timedelta(days=random.randint(0, 30), 
                                                  hours=random.randint(0, 23),
                                                  minutes=random.randint(0, 59)),
                'direction': direction,
                'thread_id': f'thread_{random.randint(1, count//2)}',
                'labels': random.sample(['INBOX', 'IMPORTANT', 'STARRED', 'SENT', 'DRAFT'], 
                                      random.randint(1, 3)),
                'attachments': random.choice([[], ['document.pdf'], ['image.jpg', 'report.xlsx']])
            }
            emails.append(email)
        
        return sorted(emails, key=lambda x: x['date'], reverse=True)
    
    @staticmethod
    def generate_social_media_posts(count: int = 10, platforms: List[str] = None) -> List[Dict[str, Any]]:
        """Generate realistic social media posts for testing"""
        if platforms is None:
            platforms = ['twitter', 'linkedin', 'facebook', 'instagram']
        
        twitter_posts = [
            "Just shipped a major update to our app! 🚀 The new features are going to make your workflow so much smoother. #productivity #tech",
            "Hot take: The future of work is asynchronous collaboration. Remote teams that master this will dominate. 🌍 #remotework #future",
            "Currently debugging at 2 AM and loving every minute of it. There's something magical about solving complex problems in the quiet hours. 💻 #coding",
            "Attended an amazing conference today. The insights on AI and machine learning were mind-blowing! 🤖 #AI #conference #learning",
            "Coffee + Code + Good Music = Perfect Morning ☕ What's your ideal working setup? #programming #lifestyle"
        ]
        
        linkedin_posts = [
            "Excited to announce that our team has successfully completed the Q3 project ahead of schedule! This achievement wouldn't have been possible without the dedication and collaboration of every team member. Here are the key factors that contributed to our success...",
            "I've been reflecting on the importance of continuous learning in today's rapidly evolving tech landscape. Over the past year, I've dedicated time to mastering new skills in cloud computing and data analysis. The investment has already paid dividends in my daily work...",
            "Just returned from an incredible industry summit where I had the opportunity to connect with leaders from various sectors. The conversations around digital transformation and sustainable business practices were particularly enlightening...",
            "Today marks my 5th anniversary at this amazing company! It's been a journey filled with growth, challenges, and incredible achievements. I'm grateful for the opportunities to lead innovative projects and work with such talented colleagues...",
            "Sharing some insights from our recent product launch. The market response has been overwhelmingly positive, and we've learned valuable lessons about customer engagement and product-market fit. Here's what worked well and what we'd do differently..."
        ]
        
        posts = []
        for i in range(count):
            platform = random.choice(platforms)
            
            if platform == 'twitter':
                text = random.choice(twitter_posts)
                metrics = {
                    'like_count': random.randint(5, 200),
                    'retweet_count': random.randint(0, 50),
                    'reply_count': random.randint(0, 25),
                    'impression_count': random.randint(500, 5000)
                }
            elif platform == 'linkedin':
                text = random.choice(linkedin_posts)
                metrics = {
                    'likes': random.randint(10, 100),
                    'comments': random.randint(2, 20),
                    'shares': random.randint(1, 15)
                }
            else:
                text = random.choice(twitter_posts + linkedin_posts)
                metrics = {
                    'likes': random.randint(5, 150),
                    'comments': random.randint(0, 30),
                    'shares': random.randint(0, 20)
                }
            
            post = {
                'id': f'{platform}_post_{i+1}_{uuid.uuid4().hex[:8]}',
                'text': text,
                'author': 'test_user',
                'author_id': f'user_{uuid.uuid4().hex[:8]}',
                'created_at': datetime.now() - timedelta(days=random.randint(0, 30),
                                                        hours=random.randint(0, 23)),
                'platform': platform,
                'url': f'https://{platform}.com/test_user/post/{i+1}',
                **metrics
            }
            
            # Add platform-specific fields
            if platform == 'twitter':
                post['public_metrics'] = {
                    'like_count': metrics['like_count'],
                    'retweet_count': metrics['retweet_count'],
                    'reply_count': metrics['reply_count'],
                    'impression_count': metrics['impression_count']
                }
            
            posts.append(post)
        
        return sorted(posts, key=lambda x: x['created_at'], reverse=True)
    
    @staticmethod
    def generate_drive_files(count: int = 10) -> List[Dict[str, Any]]:
        """Generate realistic Google Drive file data for testing"""
        file_types = [
            ('Project Proposal.pdf', 'application/pdf', 2048000),
            ('Meeting Notes.docx', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 512000),
            ('Budget Spreadsheet.xlsx', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 256000),
            ('Presentation.pptx', 'application/vnd.openxmlformats-officedocument.presentationml.presentation', 1024000),
            ('Research Data.csv', 'text/csv', 128000),
            ('Team Photo.jpg', 'image/jpeg', 3072000),
            ('Design Mockup.png', 'image/png', 1536000),
            ('Project Timeline', 'application/vnd.google-apps.document', 0),
            ('Analysis Report', 'application/vnd.google-apps.document', 0),
            ('Contact List', 'application/vnd.google-apps.spreadsheet', 0)
        ]
        
        owners = [
            ('John Smith', 'john.smith@company.com'),
            ('Sarah Johnson', 'sarah.j@example.org'),
            ('Mike Davis', 'mike.davis@tech.com'),
            ('Lisa Chen', 'l.chen@startup.io'),
            ('Test User', 'user@example.com')
        ]
        
        files = []
        for i in range(count):
            name, mime_type, size = random.choice(file_types)
            owner_name, owner_email = random.choice(owners)
            
            # Add variation to names
            if i > 0:
                name = f"{name.split('.')[0]} v{random.randint(1, 5)}.{name.split('.')[1]}" if '.' in name else f"{name} v{random.randint(1, 5)}"
            
            file_info = {
                'id': f'file_{i+1}_{uuid.uuid4().hex[:16]}',
                'name': name,
                'description': f'Important document for {random.choice(["project", "team", "client", "analysis"])}',
                'mimeType': mime_type,
                'size': str(size) if size > 0 else None,
                'createdTime': datetime.now() - timedelta(days=random.randint(1, 90)),
                'modifiedTime': datetime.now() - timedelta(days=random.randint(0, 30)),
                'owners': [{'displayName': owner_name, 'emailAddress': owner_email}],
                'shared': random.choice([True, False]),
                'webViewLink': f'https://drive.google.com/file/d/file_{i+1}/view',
                'permissions': random.choice(['reader', 'writer', 'owner'])
            }
            
            if file_info['shared']:
                file_info['webContentLink'] = f'https://drive.google.com/file/d/file_{i+1}/download'
            
            files.append(file_info)
        
        return sorted(files, key=lambda x: x['modifiedTime'], reverse=True)
    
    @staticmethod
    def generate_project_data(count: int = 5) -> List[Dict[str, Any]]:
        """Generate realistic project data for testing"""
        project_names = [
            "Project Alpha - Product Launch",
            "Customer Portal Redesign",
            "Data Migration Initiative", 
            "Mobile App Development",
            "Security Audit & Compliance",
            "Marketing Campaign Q4",
            "Infrastructure Upgrade",
            "User Experience Research",
            "Content Management System",
            "Business Intelligence Dashboard"
        ]
        
        statuses = ['planning', 'active', 'on_hold', 'completed', 'cancelled']
        priorities = ['low', 'medium', 'high', 'urgent']
        
        team_members = [
            'john.smith@company.com',
            'sarah.j@example.org', 
            'mike.davis@tech.com',
            'lisa.chen@startup.io',
            'user@example.com'
        ]
        
        projects = []
        for i in range(count):
            name = random.choice(project_names)
            status = random.choice(statuses)
            priority = random.choice(priorities)
            
            # Generate realistic timeline based on status
            created_days_ago = random.randint(30, 365)
            if status == 'completed':
                deadline = datetime.now() - timedelta(days=random.randint(1, 30))
                updated_days_ago = random.randint(1, 30)
            elif status == 'cancelled':
                deadline = datetime.now() + timedelta(days=random.randint(-30, 30))
                updated_days_ago = random.randint(7, 60)
            else:
                deadline = datetime.now() + timedelta(days=random.randint(7, 120))
                updated_days_ago = random.randint(1, 7)
            
            project = {
                'id': f'project_{i+1}_{uuid.uuid4().hex[:8]}',
                'name': name,
                'description': f'Comprehensive {name.lower()} to improve our business operations and achieve strategic objectives.',
                'status': status,
                'priority': priority,
                'deadline': deadline,
                'created_at': datetime.now() - timedelta(days=created_days_ago),
                'updated_at': datetime.now() - timedelta(days=updated_days_ago),
                'tags': random.sample(['development', 'design', 'research', 'marketing', 
                                     'infrastructure', 'security', 'data', 'mobile'], 
                                    random.randint(2, 4)),
                'team_members': random.sample(team_members, random.randint(2, 4)),
                'progress': random.randint(0, 100) / 100.0,
                'budget': random.randint(10000, 500000),
                'milestones': []
            }
            
            # Generate milestones
            milestone_count = random.randint(3, 6)
            for j in range(milestone_count):
                milestone_completed = j < (milestone_count * project['progress'])
                milestone = {
                    'id': f'milestone_{j+1}',
                    'name': f'Milestone {j+1}: {random.choice(["Planning", "Development", "Testing", "Review", "Deployment", "Launch"])}',
                    'completed': milestone_completed,
                    'due_date': project['created_at'] + timedelta(days=30 * (j+1))
                }
                
                if milestone_completed:
                    milestone['completed_at'] = milestone['due_date'] - timedelta(days=random.randint(0, 5))
                
                project['milestones'].append(milestone)
            
            projects.append(project)
        
        return sorted(projects, key=lambda x: x['updated_at'], reverse=True)
    
    @staticmethod
    def generate_personal_metrics(count: int = 30) -> List[Dict[str, Any]]:
        """Generate realistic personal metrics data for testing"""
        metric_types = ['productivity', 'habits', 'goals', 'mood', 'energy']
        
        metrics = []
        for i in range(count):
            date = datetime.now() - timedelta(days=i)
            metric_type = random.choice(metric_types)
            
            base_metric = {
                'id': f'metric_{i+1}_{uuid.uuid4().hex[:8]}',
                'date': date,
                'type': metric_type,
                'recorded_at': date + timedelta(hours=random.randint(18, 23))  # Usually recorded in evening
            }
            
            if metric_type == 'productivity':
                base_metric.update({
                    'hours_worked': random.randint(6, 12),
                    'tasks_completed': random.randint(3, 15),
                    'focus_score': random.randint(1, 10),
                    'interruptions': random.randint(0, 8),
                    'deep_work_hours': random.randint(2, 6)
                })
            elif metric_type == 'habits':
                base_metric.update({
                    'exercise': random.choice([True, False]),
                    'meditation': random.choice([True, False]),
                    'reading': random.choice([True, False]),
                    'early_wake': random.choice([True, False]),
                    'healthy_eating': random.choice([True, False])
                })
            elif metric_type == 'goals':
                base_metric.update({
                    'daily_goal_progress': random.randint(0, 100),
                    'weekly_goal_progress': random.randint(0, 100),
                    'monthly_goal_progress': random.randint(0, 100),
                    'goal_categories': random.sample(['career', 'health', 'learning', 'personal', 'financial'], 
                                                   random.randint(1, 3))
                })
            elif metric_type == 'mood':
                base_metric.update({
                    'mood_score': random.randint(1, 10),
                    'stress_level': random.randint(1, 10),
                    'energy_level': random.randint(1, 10),
                    'satisfaction_score': random.randint(1, 10),
                    'notes': random.choice([
                        "Great day overall, felt very productive",
                        "Bit stressed but got through important tasks",
                        "Low energy day, took it easy",
                        "Excellent mood, everything went smoothly",
                        "Challenging day but learned a lot"
                    ])
                })
            elif metric_type == 'energy':
                base_metric.update({
                    'morning_energy': random.randint(1, 10),
                    'afternoon_energy': random.randint(1, 10),
                    'evening_energy': random.randint(1, 10),
                    'sleep_hours': random.randint(5, 9),
                    'sleep_quality': random.randint(1, 10),
                    'caffeine_intake': random.randint(0, 5)
                })
            
            metrics.append(base_metric)
        
        return sorted(metrics, key=lambda x: x['date'], reverse=True)


# Predefined test data sets for consistent testing
SAMPLE_EMAIL_THREAD = [
    {
        'id': 'email_thread_1',
        'subject': 'Project Alpha Discussion',
        'from': 'alice@company.com',
        'to': ['bob@company.com'],
        'body': 'Hi Bob, I wanted to discuss the timeline for Project Alpha. Can we meet this week?',
        'date': datetime.now() - timedelta(days=3),
        'direction': 'received',
        'thread_id': 'thread_alpha_1'
    },
    {
        'id': 'email_thread_2', 
        'subject': 'Re: Project Alpha Discussion',
        'from': 'bob@company.com',
        'to': ['alice@company.com'],
        'body': 'Hi Alice, absolutely! I\'m free Thursday afternoon. How does 2 PM work for you?',
        'date': datetime.now() - timedelta(days=3, hours=2),
        'direction': 'sent',
        'thread_id': 'thread_alpha_1'
    },
    {
        'id': 'email_thread_3',
        'subject': 'Re: Project Alpha Discussion', 
        'from': 'alice@company.com',
        'to': ['bob@company.com'],
        'body': 'Perfect! Thursday at 2 PM works great. I\'ll send a calendar invite.',
        'date': datetime.now() - timedelta(days=2, hours=22),
        'direction': 'received',
        'thread_id': 'thread_alpha_1'
    }
]

SAMPLE_NETWORK_CONTACTS = {
    'alice@company.com': {
        'name': 'Alice Johnson',
        'interactions': 25,
        'relationship_strength': 0.8,
        'response_time_avg': 2.5  # hours
    },
    'bob@company.com': {
        'name': 'Bob Smith', 
        'interactions': 18,
        'relationship_strength': 0.7,
        'response_time_avg': 4.2
    },
    'charlie@company.com': {
        'name': 'Charlie Brown',
        'interactions': 12,
        'relationship_strength': 0.6,
        'response_time_avg': 8.1
    }
}

SAMPLE_SEARCH_RESULTS = [
    {
        'id': 'search_result_1',
        'type': 'email',
        'title': 'Project Alpha Status Update',
        'snippet': 'Latest progress on Project Alpha development...',
        'relevance_score': 0.92,
        'source': 'gmail',
        'timestamp': datetime.now() - timedelta(days=1)
    },
    {
        'id': 'search_result_2',
        'type': 'document',
        'title': 'Project Alpha Proposal.pdf',
        'snippet': 'Comprehensive proposal document for Project Alpha initiative...',
        'relevance_score': 0.88,
        'source': 'drive',
        'timestamp': datetime.now() - timedelta(days=5)
    },
    {
        'id': 'search_result_3',
        'type': 'social_post',
        'title': 'Tweet about Project Alpha',
        'snippet': 'Excited to share updates on our Project Alpha development...',
        'relevance_score': 0.75,
        'source': 'twitter',
        'timestamp': datetime.now() - timedelta(hours=6)
    }
]