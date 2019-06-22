---
title: Django vs Flask in the context of Deploying ML/AI
tags: Tech_Writings ML Django Flask Deploy
---

Django and Flask are both frameworks for creating web applications in Python. Django provides a full-featured MVC Framework that includes the whole kitchen sink. While Django alone could be used to make a RESTful API, Django REST Framework is a fantastic, feature-filled extension to the Django framework.

Flask is a micro-framework that follows the Unix philosophy of “do one thing and do it well”. Flask provides very little upfront, not even an ORM, but the community provides a large set of extensions that match a lot of Django’s feature set.

A popular idiom to compare the two frameworks is that ‘Pirates use Flask, The Navy uses Django.‘

![](/assets/blog_photos/Tech_Writings/django-vs-flask.jpg) 

# What They Have In Common

While Django and Flask take different approaches to designing a web platform, they support a lot of the same capabilities when it comes to REST API design. This analysis assumes Django is using the Django REST Framework toolkit, and Flask is using various popular plugins necessary to achieve comparable results.

## User Auth

Django REST Framework supports using Django’s built-in user model for API authentication and authorization (docs). With Flask, you can use built-in tools for basic auth, or third-party plugin such as Flask-HTTPAuth for more complex scenarios. Miquel Grinberg provides a thorough tutorial for creating API endpoints using Flask-HTTPAuth.

## Rate Limiting

Django REST framework supports throttling of both anonymous and registered users (docs). Flask, along with the Flask-Limiter plugin, support a similar feature set. Both platforms support in-memory, Memcached, or Redis backends for store rate limitation data.

## Relational Database Mapping

Django REST Framework provides a straightforward method of mapping Django ORM models to API endpoints (docs). Flask, along with Flask-Restless, provides similar capabilities for an SQLAlchemy database model (docs). Both frameworks provide a simple path to creating a CRUD interface for an existing data store.

# Advantages of Django with Django REST Framework

## Versioning

Handling multiple versions of an API is never a simple endeavor. Django REST Framework includes support for versioning is fairly flexible, making the task a little less onerous. The toolkit supports multiple URL formats (example.com/v1/sample/, v1.example.com/sample/, or example.com/sample/?version=v1), and the version is passed in as a request parameter (request.version) to be used in a view.

Versioning can be handled in Flask, but it has to be managed by the developers. In fact, many solutions involve structuring the entire hierarchy of the application to support the concept of versioning.

## Browsable API

Django REST Framework generates HTML pages to browse and execute all API endpoints. With this feature, users and/or developers can execute GETs and POSTs quickly and easily in their browser. Even if disabled for production use, this is a fantastic feature for testing and demos.

Flask does not have a good Browseable API option. The creator of Django REST Framework created a similar library for Flask called Flask-API, but it is not under active development nor is it ready for production use. Another option is to use the swagger API wrapper in front of the API, but maintaining the swagger schema would take time.

## Regular Releases

Django and Django REST Framework are both under heavy active development. Django has a major release twice per year with an LTS release every two years. Django REST Framework has had three major releases in the last year. Flask, on the other hand, last had a tagged release in 2013. This does not mean the project is dead, as the multitude of plugin libraries receive regular updates, but the contrast with Django is noteworthy.

##Community Help

This is a graph of total questions asked on Django/Flask in stackoverflow.

![](/assets/blog_photos/Tech_Writings/stack_overflow_flask_django.png) 

# Advantages of Flask

## Speed

Flask is able to achieve generally faster performance than Django(without any specific enhancements, Django can also be customized for handling multiple requests and efficiency), generally due to the fact that Flask is much more minimal in design. While both of these frameworks can support several hundred queries per second without breaking a sweat, neither were designed with a priority on speed(As our application focuses mostly on DL, there will be fewer requests but the requests themselves will be computationally expensive, so this does not hold that much). Falcon and Bottle are Python web frameworks that emphasize speed, albeit with much smaller developer communities, and are worth a look if speed is more important than support and features.

## NoSQL Support (Not a major thing)

The Django platform heavily ties itself to relational database integration. An extension called django-rest-framework-mongoengine wires Mongo into Django REST Framework, but I can’t speak for its ease of use. Because Flask has no built-in ORM, it is more freely able to integrate with NoSQL databases like MongoDB (via mongokit or Flask-Pymongo) and DynamoDB (via Flask-Dynamo). In addition, the Eve toolkit, a wrapper on top of Flask, includes built-in support for MongoDB.

Instagram, Disqus, Bitbucket, Websites of Onion, Firefox, NASA, The Washington Post use Django. 

While Pinterest uses Flask. 

# Personal Experience 

Django has a far steeper learning curve than Flask and is more complex and intricate, but in the long run, it has the flexibilities to scale with more users/intricate features. Flask is far easier to start (You could go from zero knowledge on flask to deploying a DL model on Flask for some specific case, i.e. a finished end product, in about a week of hard work), especially it’s easier to host and serve. But even if Django is harder, it has loads of flexibility and has a very strong community support, having loads of apps built onto it for specific use cases (like django-imagekit for example, the GitHub repo mentioned in the mail). 

# Why Tensorflow, why not PyTorch? 

![](/assets/blog_photos/Tech_Writings/pytorch_tensorflow.png) 

# Useful Links

These are some helpful links for further research. 

1. [Django Documentation](https://docs.djangoproject.com/en/2.2/topics/)
2. [Flask Documentation](http://flask.pocoo.org/docs/1.0/)
3. [Tensorflow Documentation](https://www.tensorflow.org/api_docs/python/tf)
4. [Keras Documentation](https://keras.io/)
5. [Python Web Frameworks](https://wiki.python.org/moin/WebFrameworks)
6. [Pytorch Vs Tensorflow](https://towardsdatascience.com/pytorch-vs-tensorflow-spotting-the-difference-25c75777377b) 
7. [ML Model to Production](https://www.linode.com/docs/applications/big-data/how-to-move-machine-learning-model-to-production/) 
8. [Deploying a DL Model ](https://medium.com/@maheshkkumar/a-guide-to-deploying-machine-deep-learning-model-s-in-production-e497fd4b734a) 
9. [ML as APIs using Flask](https://www.analyticsvidhya.com/blog/2017/09/machine-learning-models-as-apis-using-flask/) 
10. [Django vsFlask](https://www.educba.com/django-vs-flask/) 
11. [Restful API - Django vs Flask](https://www.excella.com/insights/creating-a-restful-api-django-rest-framework-vs-flask)
12. [Depolying a ML model as a RestAPI](https://towardsdatascience.com/deploying-a-machine-learning-model-as-a-rest-api-4a03b865c166) 