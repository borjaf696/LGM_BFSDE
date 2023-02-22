## Notas reunión Álvaro 20230119

Fácil y sencillo al final resulta que el $D(T)$ representaba $d(t)$ y venía a decir que tenía que ser posible recuperar el valor de descuento en cualquier punto de la simulación.

#### Notas reunión

* Simplificar simulaciones:
    * T = 1 (done)
    * N_steps = 15 (done)
* Checkear el gradiente (es vector).
* Cambiar el payoff a vencimiento (escalar por el numerario).
* Checkear derivadas que devuelve el gradient tape
* ***Si sigue fallando pasar la derivada analítica de $Z(\cdot)$***

#### Dudas

Hablado dos reuniones atrás con Francisco. Ahora mismo tenemos una arquitecture seq2seq, esto implica recibimos una secuencia y devolvemos otra de igual tamaño. De cara a la red de neuronas implica que tenemos un $X = [X_1, \dots, _n]$ y devolvemos un $Y = [Y_1, \dots, Y_n]$ donde $Y = f(W\cdot X + b)$ en el caso de una sola capa. Por lo tanto, $\nabla f = ...$ (matriz)

Problema:

* El gradiente de la red dada la entrada en una matriz, esto toca la función de pérdida que compara la derivada de $\phi(\cdot)$ con la derivada de la red a vencimiento. ¿Como comparamos la matriz con un número? 
    * Solución rápida: 
        * Coger $\frac{\delta NN_n}{\delta x_n}$ para la función de pérdida.
        * Coger la diagonal de la Jacobiana para la estimación de V
        * Conceptualmente, ¿Es esto correcto? 
        * Recordatorio:
        $$\mathcal{L}(\overline{V}, \hat{V}) = \beta_1 \cdot (\hat{V}_n - \phi(n, x_n))^2 + \beta_2\cdot (\frac{\delta \hat{V}_n}{\delta x_n} - \frac{\delta F(X)}{\delta x_n})^2 + \sum_{i = 1}^{n - 1}(\overline{V}_i - \hat{V}_i)^2$$
* La alternativa es usar N_steps redes de neuronas que reciban un X y ajusten $f(X)$:
    * Predecir con N_steps redes.
        * Usar el primero para calcular V
    * Calcular la función de pérdida, y aquí viene el problema como calculamos el gradiente para los pesos de las redes dado el loss que tenemos. Al estar en distintas arquitecturas aquí se complica la cosa, ¿no? 
        * Esto tengo que mirarlo porque igual si se puede hacer. Si hacemos las predicciones de todas las redes en un GradientTape igual si podemos calcular de manera sencilla la pérdida e incorporarla. Intento hacerlo para antes del miércoles.
* Una opción C es eliminar el término $\sum_{i = 1}^{n - 1}(\overline{V}_i - \hat{V}_i)^2$ y cambiarlo por $\overline{V}_0 - \hat{V}_0$. Esto simplificaría mucho todo.