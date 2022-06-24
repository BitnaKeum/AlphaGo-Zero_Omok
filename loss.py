# Cross Entropy loss function으로 전달하기 전에 잘못된 움직임으로부터 예측을 마스킹하는 사용자 정의 손실 함수가 포함되어 있음

# tensorflow 말고 다른 라이브러리로 대체 가능

import tensorflow as tf

def softmax_cross_entropy_with_logits(y_true, y_pred):

	p = y_pred
	pi = y_true

	zero = tf.zeros(shape=tf.shape(pi), dtype=tf.float32)
	where = tf.equal(pi, zero)

	negatives = tf.fill(tf.shape(pi), -100.0) 
	p = tf.where(where, negatives, p)

	loss = tf.nn.softmax_cross_entropy_with_logits(labels=pi, logits=p)

	return loss
