"""Package the current authored project, including uncommitted course updates.

Explicitly excludes Git metadata, environments, credentials and runtime weights.
This creates a local artifact only; it does not upload or deploy anything.
"""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED

ROOT=Path(__file__).resolve().parents[1]
TOP=['README.md','CONTRIBUTING.md','package.json','.gitignore','.github/workflows/checks.yml']
DIRECTORIES=['docs','labs','scripts','tests','web','reports']
ALLOWED={'.md','.py','.mjs','.json','.html','.css','.svg','.jpg','.png','.yml','.txt','.zip'}


def package():
    files=[ROOT/name for name in TOP if (ROOT/name).is_file()]
    for directory in DIRECTORIES:
        for path in (ROOT/directory).rglob('*'):
            if path.is_file() and path.suffix in ALLOWED and not any(p in {'.git','__pycache__','node_modules'} for p in path.parts):
                files.append(path)
    files=sorted(set(files))
    output=ROOT/'artifacts/SafePostTrain-Lab-stage1.zip'
    output.parent.mkdir(exist_ok=True)
    manifest={}
    with ZipFile(output,'w',compression=ZIP_DEFLATED) as archive:
        for path in files:
            name=str(path.relative_to(ROOT));content=path.read_bytes()
            manifest[name]={'sha256':hashlib.sha256(content).hexdigest(),'bytes':len(content)}
            info=ZipInfo('SafePostTrain-Lab/'+name,date_time=(2026,1,1,0,0,0));info.external_attr=0o100644<<16;info.compress_type=ZIP_DEFLATED
            archive.writestr(info,content)
        info=ZipInfo('SafePostTrain-Lab/SOURCE_MANIFEST.json',date_time=(2026,1,1,0,0,0));info.compress_type=ZIP_DEFLATED
        archive.writestr(info,json.dumps(manifest,ensure_ascii=False,indent=2))
    with ZipFile(output) as archive:
        assert archive.testzip() is None
        for name in archive.namelist():
            assert '.site-deploy' not in name and '.venv' not in name and '/.git/' not in name and '/.env' not in name
    print(json.dumps({'archive':str(output),'files':len(files),'bytes':output.stat().st_size},ensure_ascii=False))
    return output


if __name__=='__main__':
    package()
