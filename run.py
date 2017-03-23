from flask import *
from genetic.genetic import GeneticAlgorithm, Labyrinth
from pathlib import Path
from multiprocessing import Process
from bokeh.embed import components
from bokeh.plotting import figure
import numpy as np

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        with open('count.txt', 'r+') as f:
            i = f.readline()
            # TODO: Optimize this?
            if i is '':
                i = 0
            else:
                i = int(i) + 1
            f.seek(0)
            f.write(str(i))
            f.truncate()
        p1 = Process(target=GeneticAlgorithm(
            labyrinth=Labyrinth(file_obj=request.files.get(request.form.get('labyrinth'))),
            num_population=request.form.get('pop'),
            max_iter=request.form.get('iters'),
            crossover_rate=request.form.get('crossover_rate'),
            crossover_pts=request.form.get('crossover_pts'),
            mutation_rate=request.form.get('mutation_rate'),
            selection=request.form.get('selection'),
            elitism_num=request.form.get('elites'),
            min_moves_mult=request.form.get('min_moves_mult'),
            max_moves_mult=request.form.get('max_moves_mult'),
            file_name='{}'.format(i)).save_data(to_html=False, plot_dir='static/plots/',
                                                save_image=True, file_dir='static/arrays/'))
        p1.start()
        return redirect('/{}'.format(i))

    else:
        return render_template('index.html')


@app.route('/<plot_id>', methods=['GET'])
def show_plots(plot_id):
    if Path('static/arrays/{}.npy'.format(plot_id)).is_file():
        pop, max_gen, winner_moveset, labyrinth, selection, avg_fitness, max_iter, setup = np.load(
            '{}.npy'.format(plot_id))

        # Last pop fitness plot
        s1 = figure(width=500, plot_height=500, title='Fitness of the last population')
        s1.line(np.arange(len(pop[:, 1])), pop[:, 1])
        script1, div1 = components(s1)

        # Average fitness plot
        s2 = figure(width=500, height=500, title='Average fitness (before making it positive)')
        s2.line(np.arange(len(avg_fitness)), avg_fitness)
        script2, div2 = components(s2)

        return render_template('plots.html'.format(plot_id), script=script1,
                               div=div2, script2=script2, div2=div1,
                               plot_url=url_for('static', filename='plots/{}.png'.format(plot_id)), id=plot_id,
                               setup=setup,
                               winner_moveset=zip(str(winner_moveset).split(' '),
                                                  winner_moveset.move_string_pairs))
    else:
        return render_template_string('Wait for the process to finish')


if __name__ == '__main__':
    app.run(debug=True)
