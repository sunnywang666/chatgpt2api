"""A completed image request must retain every returned output before archive."""
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import pytest
from PIL import Image

spec=importlib.util.spec_from_file_location('bounded_c17_round',Path(__file__).parents[1]/'scripts/acceptance/bounded_c17_round.py')
client=importlib.util.module_from_spec(spec);spec.loader.exec_module(client)
driver_spec=importlib.util.spec_from_file_location('real_candidate_client',Path(__file__).parents[1]/'scripts/acceptance/real_candidate_client.py')
real_client=importlib.util.module_from_spec(driver_spec);driver_spec.loader.exec_module(real_client)


def test_text_recovery_reads_existing_result_without_overwriting(tmp_path):
    receipt={'status':'succeeded','content':'original result'}
    record={'id':'original'}
    real_client._save_text_content(record,receipt,tmp_path)
    target=tmp_path/'original.txt';before=target.stat().st_mtime_ns
    recovered={'id':'original'}
    real_client._save_text_content(recovered,receipt,tmp_path)
    assert recovered==record and target.stat().st_mtime_ns==before
    with pytest.raises(SystemExit):
        real_client._save_text_content({'id':'original'},{'status':'succeeded','content':'changed'},tmp_path)
    assert target.read_text()=='original result'

def png(color):
    out=BytesIO();Image.new('RGB',(2,3),color).save(out,format='PNG');return out.getvalue()

@pytest.mark.parametrize('broken_second',[False,True])
def test_all_returned_images_are_saved_and_budgeted(tmp_path,broken_second):
    payloads=[png('red'),b'not an image' if broken_second else png('blue')]
    reads=[];halts=[]
    def open_result(method,url):
        index=int(url.rsplit('/',1)[1]);reads.append(index);return BytesIO(payloads[index])
    driver=SimpleNamespace(_image_receipt=lambda *a:{'data':[{},{}]},_save=lambda *a:None,
                           _halt=lambda *a:halts.append(a[-1]))
    record={'id':'original','budget_reserved':{'image':1}}
    ok=client.save_image(SimpleNamespace(open=open_result),SimpleNamespace(MAX_DOWNLOAD_BYTES=4096),driver,tmp_path/'state',{},record,tmp_path)
    assert reads==[0,1]
    assert (tmp_path/'original.png').read_bytes()==payloads[0]
    if broken_second:
        assert not ok and halts==['IMAGE_RESULT_VALIDATION_FAILED']
        assert 'actual_image_count' not in record
    else:
        assert ok and not halts
        assert (tmp_path/'original-1.png').read_bytes()==payloads[1]
        assert record['actual_image_count']==record['budget_reserved']['image']==2
        assert len(record['result_files'])==2
        assert client.save_image(SimpleNamespace(open=open_result),SimpleNamespace(MAX_DOWNLOAD_BYTES=4096),driver,tmp_path/'state',{},record,tmp_path)
        assert record['budget_reserved']['image']==2
