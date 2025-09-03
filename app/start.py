import os

import uvicorn

from app.migrate import run_migrations

if __name__ == '__main__':
    run_migrations()
    uvicorn.run(
        'app.main:app',
        host='0.0.0.0',
        port=int(os.getenv('PORT', '8000')),
    )
